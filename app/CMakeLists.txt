# ACE15-Realtime Standalone app (JUCE 8). Standalone first; AU/VST3 later.
#
# Build:
#   cmake -S app -B app/build -DCMAKE_BUILD_TYPE=Release
#   cmake --build app/build -j
# Run:
#   app/build/ACE15_artefacts/Release/Standalone/ACE15.app/Contents/MacOS/ACE15
cmake_minimum_required(VERSION 3.22)

if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.5" CACHE STRING "")
endif()

project(ACE15RealtimeApp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
    JUCE
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        8.0.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(ACE15
    PRODUCT_NAME              "ACE15 Realtime"
    COMPANY_NAME              "MaxGraf"
    BUNDLE_ID                 "com.maxgraf.ace15realtime"
    PLUGIN_MANUFACTURER_CODE  "Maxg"
    PLUGIN_CODE               "Ac15"
    FORMATS                   Standalone
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    IS_MIDI_EFFECT            FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS FALSE
)

juce_generate_juce_header(ACE15)

juce_add_binary_data(ACE15Resources SOURCES
    Resources/index.html
    Resources/app.js
    Resources/styles.css
)

target_sources(ACE15 PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/IpcClient.cpp
)

target_compile_definitions(ACE15 PUBLIC
    JUCE_WEB_BROWSER=1
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(ACE15 PRIVATE
    ACE15Resources
    juce::juce_audio_utils
    juce::juce_gui_extra
    PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags
)
