// Minimal end-to-end example: fetch pre-generated training artifacts +
// synthetic data (produced offline by scripts/generate_artifacts.py and
// scripts/make_synthetic_data.py, same as the native CLI's toy example),
// run training entirely in-browser, and offer the fine-tuned model as a
// download. No bundler -- open index.html through a local static server
// (needs COOP/COEP headers if you turn on wasm threads; the module here
// builds with threads OFF, so a plain static server is enough).

const INPUT_DIM = 4;
const TARGET_DIM = 1;
const BATCH_SIZE = 32;
const EPOCHS = 20;
const LEARNING_RATE = 0.01;

async function fetchBytes(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`fetch ${path} failed: ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

function log(msg) {
  const el = document.getElementById('log');
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

async function main() {
  log('loading wasm module...');
  const Module = await createOnnxFinetune();

  log('fetching training artifacts + synthetic data...');
  const [checkpoint, trainingModel, evalModel, optimizerModel, inputBytes, targetBytes] =
    await Promise.all([
      fetchBytes('artifacts/checkpoint'),
      fetchBytes('artifacts/training_model.onnx'),
      fetchBytes('artifacts/eval_model.onnx'),
      fetchBytes('artifacts/optimizer_model.onnx'),
      fetchBytes('train_input.bin'),
      fetchBytes('train_target.bin'),
    ]);

  const inputs = new Float32Array(inputBytes.buffer, inputBytes.byteOffset, inputBytes.length / 4);
  const targets = new Float32Array(targetBytes.buffer, targetBytes.byteOffset, targetBytes.length / 4);
  const numSamples = inputs.length / INPUT_DIM;

  log(`building training session (${numSamples} samples)...`);
  const session = new Module.FinetuneSession(checkpoint, trainingModel, evalModel, optimizerModel);
  session.setLearningRate(LEARNING_RATE);

  const order = Array.from({ length: numSamples }, (_, i) => i);

  let step = 0;
  for (let epoch = 0; epoch < EPOCHS; ++epoch) {
    // Fisher-Yates shuffle each epoch, same as the native CLI.
    for (let i = order.length - 1; i > 0; --i) {
      const j = Math.floor(Math.random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    for (let start = 0; start + BATCH_SIZE <= numSamples; start += BATCH_SIZE) {
      const batchInput = new Float32Array(BATCH_SIZE * INPUT_DIM);
      const batchTarget = new Float32Array(BATCH_SIZE * TARGET_DIM);
      for (let b = 0; b < BATCH_SIZE; ++b) {
        const src = order[start + b];
        batchInput.set(inputs.subarray(src * INPUT_DIM, (src + 1) * INPUT_DIM), b * INPUT_DIM);
        batchTarget.set(targets.subarray(src * TARGET_DIM, (src + 1) * TARGET_DIM), b * TARGET_DIM);
      }

      const loss = session.trainStep(batchInput, batchTarget, BATCH_SIZE, INPUT_DIM, TARGET_DIM);
      session.optimizerStep();
      session.lazyResetGrad();

      if (step % 10 === 0) log(`epoch ${epoch} step ${step} loss ${loss.toFixed(6)}`);
      ++step;
    }
  }

  log('exporting fine-tuned model...');
  const finetuned = session.exportModel(['output']);

  const blob = new Blob([finetuned], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.getElementById('download');
  a.href = url;
  a.download = 'finetuned.onnx';
  a.textContent = `download finetuned.onnx (${finetuned.length} bytes)`;
  a.style.display = 'inline';

  log('done.');
}

main().catch((err) => log('ERROR: ' + err.message));
