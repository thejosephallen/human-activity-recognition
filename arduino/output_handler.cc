/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)
#define ARDUINO_EXCLUDE_CODE
#endif  // defined(ARDUINO) && !defined(ARDUINO_ARDUINO_NANO33BLE)

#include "output_handler.h"

#include "Arduino.h"

void HandleOutput(tflite::ErrorReporter* error_reporter, int kind) {
  // The first time this method runs, set up our LED
  static bool is_initialized = false;
  if (!is_initialized) {
    pinMode(LED_BUILTIN, OUTPUT);
    is_initialized = true;
  }
  // light (red: wing, blue: ring, green: slope)
  if (kind == 0) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "SIT/STAND\n\r");
  } else if (kind == 1) {
    digitalWrite(LED_BUILTIN, HIGH); //blue
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "WALK\n\r");
  } else if (kind == 2) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "LIEDOWN\n\r");
  } else if (kind == 3) {
    digitalWrite(LED_BUILTIN, LOW); //green 
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "JUMP\n\r");
  } else if (kind == 4) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "TRANSITION/UNDEFINED\n\r");
  } else if (kind == 5) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "STAIRSUP\n\r");
  } else if (kind == 6) {
    TF_LITE_REPORT_ERROR(
        error_reporter,
        "STAIRSDOWN\n\r");
  } 
}