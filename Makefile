TARGET  = eigensolver
CC      = g++
CFLAGS  = -O2 -Wall -std=c++11 -march=native -DNDEBUG -fopenmp -DEIGEN_DONT_PARALLELIZE # -g -ggdb 

LINKER  = $(CC) -o
LFLAGS  = $(CFLAGS) 

SRC     = src
LIB     = lib
BUILD   = build

CFLAGS	+=           \
	-I$(SRC)         \
	-I$(LIB)         \
	-I$(LIB)/eigen3

SOURCES := $(wildcard $(SRC)/*.cpp)
OBJECTS := $(SOURCES:$(SRC)/%.cpp=$(BUILD)/%.o)
rm      = rm -rf


$(BUILD)/$(TARGET): $(OBJECTS)
	@$(LINKER) $@ $(LFLAGS) $(OBJECTS)

$(OBJECTS): $(BUILD)/%.o : $(SRC)/%.cpp
	@mkdir -p $(BUILD)
	@$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	@$(rm) $(BUILD)

