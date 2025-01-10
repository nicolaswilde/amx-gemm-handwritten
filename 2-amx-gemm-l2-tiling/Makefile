CFLAG = -Ofast -march=native -fno-strict-aliasing
CC = gcc
BIN = amx-gemm
CFILES = amx-gemm.c
CORE = 1

PERFFLAGS = -e L1-dcache-loads,L1-dcache-load-misses \
            -e l2_request.all,l2_request.miss \
			-e LLC-loads,LLC-load-misses

all: compile objdump run

compile:
	$(CC) $(CFLAG) $(CFILES) -o $(BIN) $(LIBS)

lockfreq:
	echo userspace | sudo tee /sys/devices/system/cpu/cpu1/cpufreq/scaling_governor
	echo 2300000 | sudo tee /sys/devices/system/cpu/cpu1/cpufreq/scaling_setspeed

run: lockfreq
	taskset -c $(CORE) ./$(BIN)

perf: lockfreq
	taskset -c $(CORE) sudo perf stat $(PERFFLAGS) ./$(BIN)

objdump:
	objdump -d $(BIN) > $(BIN).s

clean:
	-rm $(BIN)

.PHONY: all compile run perf objdump clean