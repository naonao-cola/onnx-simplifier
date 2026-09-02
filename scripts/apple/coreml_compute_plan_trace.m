// coreml_compute_plan_trace.m
//
// Dumps a compiled Core ML model's per-operation compute-unit placement
// (CPU/GPU/ANE) and MLComputePlan's estimated relative cost, as a Chrome
// Trace Event Format JSON file (openable at chrome://tracing or
// https://ui.perfetto.dev) -- one "thread" lane per compute device, so it's
// visually obvious at a glance which ops landed where.
//
// IMPORTANT: this is a STATIC estimate, not a measurement. MLComputePlan
// analyzes a compiled model without running it -- "dur" in the emitted
// trace is MLComputePlanCost's `weight` (a relative-cost fraction, not
// microseconds) scaled by 1e6 purely so the trace viewer renders a legible
// timeline; it is not wall-clock time from an actual .predict() call. Real
// per-op timing would need Instruments' Core ML template (`xcrun xctrace
// record --template "Core ML"`) attached to an actual prediction run --
// out of scope here since that trace format is far less tractable to parse
// than MLComputePlan's structured API. This tool exists to answer a
// narrower, cheaper question: which compute unit does Core ML's own
// placement logic prefer for each op, before spending any time running it.
//
// Only implements the traversal an ML Program model (the format
// onnxsim/coreml_export.py always emits, convert_to="mlprogram") needs:
// MLComputePlan.load's documented, public, async API
// (loadContentsOfURL:configuration:completionHandler:), walking
// modelStructure.program.functions["main"].block.operations. Does not
// recurse into nested control-flow blocks (cond/while_loop) -- onnxsim's
// translator doesn't emit those ops, so every op it could produce is
// reachable at that top level.
//
// Adapted from the traversal shown in
// https://github.com/freedomtan/coreml_modelc_profling (the public
// MLComputePlan API calls -- computeDeviceUsageForMLProgramOperation:,
// estimatedCostOfMLProgramOperation: -- match that reference; the device
// object returned by -preferredComputeDevice is classified via its
// -description string rather than isKindOfClass: against
// MLCPUComputeDevice/MLGPUComputeDevice/MLNeuralEngineComputeDevice, since
// this file has no way to confirm those class names are part of the public
// header in the SDK a given CI runner has -- description string matching
// only assumes NSObject's universally-guaranteed -description, at the cost
// of falling back to "unknown" instead of failing to build if Apple ever
// renames them.
//
// Build (matches https://github.com/freedomtan/coreml_modelc_profling's
// Makefile -- plain clang, no Xcode project needed):
//   clang -O2 -o coreml_compute_plan_trace coreml_compute_plan_trace.m \
//       -framework CoreML -framework Foundation
//
// Usage (model must already be compiled -- `xcrun coremlcompiler compile
// model.mlpackage <output_dir>` first, MLComputePlan only loads .mlmodelc):
//   ./coreml_compute_plan_trace model.mlmodelc out.json [compute_units]
// compute_units: all (default) | cpuOnly | cpuAndGPU | cpuAndNeuralEngine
// -- matches this repo's existing MLComputeUnits=ALL convention
// (coreml_backend.py, run_llm_decode_benchmark.py) when left at the
// default, so the placement shown matches what the benchmark actually ran
// under.

#import <CoreML/CoreML.h>
#include <stdio.h>

typedef struct {
  NSMutableArray<NSDictionary *> *events;
  NSMutableDictionary<NSString *, NSNumber *> *cursorByLane;
  NSMutableDictionary<NSString *, NSNumber *> *opCountByLane;
  NSMutableDictionary<NSString *, NSNumber *> *weightSumByLane;
} TraceState;

static NSDictionary<NSString *, NSNumber *> *LaneTids(void) {
  return @{@"ane" : @2, @"gpu" : @1, @"cpu" : @0, @"unknown" : @3};
}

static NSString *ClassifyDevice(id device) {
  if (!device) return @"unknown";
  NSString *desc = [device description] ?: @"";
  if ([desc rangeOfString:@"Neural" options:NSCaseInsensitiveSearch].location != NSNotFound) {
    return @"ane";
  }
  if ([desc rangeOfString:@"GPU" options:NSCaseInsensitiveSearch].location != NSNotFound) {
    return @"gpu";
  }
  if ([desc rangeOfString:@"CPU" options:NSCaseInsensitiveSearch].location != NSNotFound) {
    return @"cpu";
  }
  return @"unknown";
}

static void RecordOperation(TraceState *state, NSString *lane, NSString *opName, double weight,
                            NSString *deviceDescription) {
  NSNumber *tid = LaneTids()[lane] ?: @3;
  double cursor = [state->cursorByLane[lane] ?: @0 doubleValue];
  // Scaled purely for a legible timeline -- see the module comment: this is
  // not measured wall-clock time.
  double dur = MAX(weight * 1e6, 1.0);

  [state->events addObject:@{
    @"name" : opName ?: @"?",
    @"cat" : @"coreml_compute_plan",
    @"ph" : @"X",
    @"ts" : @(cursor),
    @"dur" : @(dur),
    @"pid" : @1,
    @"tid" : tid,
    @"args" : @{
      @"estimated_cost_weight" : @(weight),
      @"preferred_device" : deviceDescription ?: @"?",
    },
  }];
  state->cursorByLane[lane] = @(cursor + dur);
  state->opCountByLane[lane] = @([state->opCountByLane[lane] ?: @0 intValue] + 1);
  state->weightSumByLane[lane] = @([state->weightSumByLane[lane] ?: @0 doubleValue] + weight);
}

static int WalkProgram(MLComputePlan *computePlan, MLModelStructureProgram *program,
                       TraceState *state) {
  if (!program) {
    fprintf(stderr, "error: model structure has no ML Program.\n");
    return 1;
  }
  MLModelStructureProgramFunction *mainFunction = program.functions[@"main"];
  if (!mainFunction) {
    fprintf(stderr, "error: ML Program has no 'main' function.\n");
    return 1;
  }

  NSArray<MLModelStructureProgramOperation *> *operations = mainFunction.block.operations;
  for (MLModelStructureProgramOperation *operation in operations) {
    MLComputePlanDeviceUsage *deviceUsage =
        [computePlan computeDeviceUsageForMLProgramOperation:operation];
    MLComputePlanCost *estimatedCost = [computePlan estimatedCostOfMLProgramOperation:operation];
    if (!deviceUsage || !estimatedCost) {
      // MLComputePlan returns nil for ops it couldn't analyze (e.g. pure
      // metadata/const ops with no runtime cost) -- skip rather than
      // recording a misleading zero-cost/unknown-device entry.
      continue;
    }
    id preferredDevice = [deviceUsage preferredComputeDevice];
    NSString *lane = ClassifyDevice(preferredDevice);
    RecordOperation(state, lane, [operation operatorName], [estimatedCost weight],
                    [preferredDevice description]);
  }
  return 0;
}

static MLComputeUnits ParseComputeUnits(const char *arg) {
  if (!arg) return MLComputeUnitsAll;
  NSString *s = [NSString stringWithUTF8String:arg];
  if ([s caseInsensitiveCompare:@"cpuOnly"] == NSOrderedSame) return MLComputeUnitsCPUOnly;
  if ([s caseInsensitiveCompare:@"cpuAndGPU"] == NSOrderedSame) return MLComputeUnitsCPUAndGPU;
  if ([s caseInsensitiveCompare:@"cpuAndNeuralEngine"] == NSOrderedSame) {
    return MLComputeUnitsCPUAndNeuralEngine;
  }
  return MLComputeUnitsAll;
}

int main(int argc, char *argv[]) {
  @autoreleasepool {
    if (argc < 3) {
      fprintf(stderr,
              "usage: %s <model.mlmodelc> <out.json> "
              "[all|cpuOnly|cpuAndGPU|cpuAndNeuralEngine]\n",
              argv[0]);
      return 2;
    }
    NSURL *modelURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]
                                 isDirectory:YES];
    NSString *outPath = [NSString stringWithUTF8String:argv[2]];
    MLComputeUnits computeUnits = ParseComputeUnits(argc > 3 ? argv[3] : NULL);

    MLModelConfiguration *configuration = [[MLModelConfiguration alloc] init];
    configuration.computeUnits = computeUnits;

    __block int exitCode = 0;
    TraceState state = {
        .events = [NSMutableArray array],
        .cursorByLane = [NSMutableDictionary dictionary],
        .opCountByLane = [NSMutableDictionary dictionary],
        .weightSumByLane = [NSMutableDictionary dictionary],
    };

    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    [MLComputePlan
        loadContentsOfURL:modelURL
            configuration:configuration
        completionHandler:^(MLComputePlan *_Nullable computePlan, NSError *_Nullable error) {
          if (!computePlan) {
            fprintf(stderr, "error: failed to load compute plan: %s\n",
                    error ? [[error localizedDescription] UTF8String] : "unknown error");
            exitCode = 1;
            dispatch_semaphore_signal(done);
            return;
          }
          MLModelStructure *modelStructure = [computePlan modelStructure];
          if (modelStructure.program) {
            exitCode = WalkProgram(computePlan, modelStructure.program, &state);
          } else if (modelStructure.neuralNetwork) {
            fprintf(stderr, "error: neuralnetwork-format models aren't supported here (only "
                            "mlprogram, what onnxsim/coreml_export.py always emits) -- "
                            "MLComputePlan has no estimatedCost API for neural network layers.\n");
            exitCode = 1;
          } else {
            fprintf(stderr, "error: unsupported or unrecognized model structure.\n");
            exitCode = 1;
          }
          dispatch_semaphore_signal(done);
        }];
    // No fixed timeout: loading the plan for a several-billion-parameter
    // model is expected to take a while; wait for the documented completion
    // handler rather than guessing a sleep duration.
    dispatch_semaphore_wait(done, DISPATCH_TIME_FOREVER);
    if (exitCode != 0) return exitCode;

    NSDictionary *trace = @{
      @"traceEvents" : state.events,
      @"displayTimeUnit" : @"ms",
      @"otherData" : @{
        @"source" : @"coreml_compute_plan_trace.m",
        @"note" : @"dur is MLComputePlan's estimated relative cost weight scaled "
                  @"for legibility, not measured wall-clock time -- see the file's "
                  @"module comment.",
        @"compute_units_requested" : [NSString stringWithUTF8String:(argc > 3 ? argv[3] : "all")],
      },
    };
    NSError *jsonError = nil;
    NSData *jsonData = [NSJSONSerialization dataWithJSONObject:trace
                                                       options:NSJSONWritingPrettyPrinted
                                                         error:&jsonError];
    if (!jsonData) {
      fprintf(stderr, "error: failed to serialize JSON: %s\n",
              [[jsonError localizedDescription] UTF8String]);
      return 1;
    }
    if (![jsonData writeToFile:outPath atomically:YES]) {
      fprintf(stderr, "error: failed to write %s\n", [outPath UTF8String]);
      return 1;
    }

    for (NSString *lane in @[ @"ane", @"gpu", @"cpu", @"unknown" ]) {
      int count = [state.opCountByLane[lane] ?: @0 intValue];
      if (count == 0) continue;
      double weightSum = [state.weightSumByLane[lane] ?: @0 doubleValue];
      printf("%-8s %4d ops, %6.2f%% of total estimated cost\n", [lane UTF8String], count,
             weightSum * 100.0);
    }
    printf("wrote %s\n", [outPath UTF8String]);
    return 0;
  }
}
