include_directories(${SDK_ROOT}/lib/nncase/v1/include)
include_directories(${SDK_ROOT}/third_party/gsl-lite/include)
include_directories(${SDK_ROOT}/third_party/mpark-variant/include)
include_directories(${SDK_ROOT}/third_party/nlohmann_json/include)
include_directories(${SDK_ROOT}/third_party/xtl/include)

# Board selection (BOARD=m5stickv|cube, set by build.sh). The SDK resets
# CMAKE_C/CXX_FLAGS, so a -D on the command line would be dropped; this hook
# is honoured.
if("$ENV{BOARD}" STREQUAL "cube")
  add_definitions(-DBOARD_CUBE)
else()
  add_definitions(-DBOARD_M5STICKV)
endif()
