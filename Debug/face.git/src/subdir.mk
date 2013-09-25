################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../face.git/src/face.cpp 

OBJS += \
./face.git/src/face.o 

CPP_DEPS += \
./face.git/src/face.d 


# Each subdirectory must supply rules for building sources it contributes
face.git/src/%.o: ../face.git/src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -I/home/jorge/stasm4.0.0/stasm/ -I/usr/local/include -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


