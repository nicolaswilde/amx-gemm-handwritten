# lock the cpu frequency for all cores
CPU_CORES ?= 240
CORE_LIST = $(shell seq 0 $(shell expr $(CPU_CORES) - 1))
FREQ ?= 2300000 # 2.3 GHz

# test cpu cores
CORES_M ?= 4
CORES_N ?= 4

# performance test type
IS_TEST_MNK ?= 1
IS_TEST_K ?= 0
TEST_M ?= 2048
TEST_N ?= 2048

CFLAG = -Ofast -march=native -fno-strict-aliasing -fopenmp \
        -DCORES_M=$(CORES_M) -DCORES_N=$(CORES_N) \
		-DIS_TEST_MNK=$(IS_TEST_MNK) -DIS_TEST_K=$(IS_TEST_K) \
		-DTEST_M=$(TEST_M) -DTEST_N=$(TEST_N)
CC = gcc
BIN = amx-gemm
CFILES = amx-gemm.c

PERFFLAGS = -e L1-dcache-loads,L1-dcache-load-misses \
            -e l2_request.all,l2_request.miss \
			-e LLC-loads,LLC-load-misses

all: compile objdump run

compile:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

lockfreq:
	for i in $(CORE_LIST); do \
		echo userspace | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
		echo $(FREQ) | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_setspeed > /dev/null; \
	done

unlockfreq:
	for i in $(CORE_LIST); do \
		echo ondemand | sudo tee /sys/devices/system/cpu/cpu$$i/cpufreq/scaling_governor > /dev/null; \
	done

run:
	make lockfreq
	./$(BIN)
	make unlockfreq

perf:
	make lockfreq
	sudo perf stat $(PERFFLAGS) ./$(BIN)
	make unlockfreq

objdump:
	objdump -d $(BIN) > $(BIN).s

clean:
	-rm $(BIN)

.PHONY: all compile lockfreq unlockfreq run perf objdump clean