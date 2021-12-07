
har_SRCS := \
main.cc \
main_functions.cc \
accelerometer_handler.cc \
gesture_predictor.cc \
output_handler.cc

har_HDRS := \
main_functions.h \
constants.h \
accelerometer_handler.h \
gesture_predictor.h \
output_handler.h

har_GENERATOR_INPUTS := \
../train/models/model.tflite

har_GENERATED_SRCS := \
$(GENERATED_SRCS_DIR)har_model_data.cc

har_GENERATED_HDRS := \
$(GENERATED_SRCS_DIR)har_model_data.h

# Builds a standalone binary
$(eval $(call microlite_test,har,\
$(har_SRCS),$(har_HDRS),$(har_GENERATOR_INPUTS)))

har_SRCS += $(har_GENERATED_SRCS)
har_HDRS += $(har_GENERATED_HDRS)

list_har_example_sources:
	@echo $(har_SRCS)

list_har_example_headers:
	@echo $(har_HDRS)