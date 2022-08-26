CXX      := g++
CXXFLAGS := -Wall -std=c++14
LDFLAGS  := -L/usr/lib -lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := program
INCLUDE  := -I./
SRC      :=                      \
   $(wildcard src/*.cpp)         \

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)
DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info

build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O3
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"

run: $(APP_DIR)/$(TARGET)
	./$(APP_DIR)/$(TARGET) fptree $(PWD)/data/webdocs.dat 98.0 $(PWD)/data/my_out_98.dat

test: $(APP_DIR)/$(TARGET)

	./$(APP_DIR)/$(TARGET) fptree $(PWD)/testcases/1/test1.dat 30.0 $(PWD)/testcases/1/my_out.dat
	diff -daZ testcases/1/out_30.dat_sorted testcases/1/my_out.dat
	
	./$(APP_DIR)/$(TARGET) fptree $(PWD)/testcases/2/test2.dat 30.0 $(PWD)/testcases/2/my_out.dat
	diff -daZ testcases/2/out_30.dat_sorted testcases/2/my_out.dat

	./$(APP_DIR)/$(TARGET) fptree $(PWD)/testcases/3/test3.dat 01.0 $(PWD)/testcases/3/my_out.dat
	diff -daZ testcases/3/out_1.dat_sorted testcases/3/my_out.dat

	./$(APP_DIR)/$(TARGET) fptree $(PWD)/testcases/4/test4.dat 90.0 $(PWD)/testcases/4/my_out.dat
	diff -daZ testcases/4/out_90.dat_sorted testcases/4/my_out.dat

	./$(APP_DIR)/$(TARGET) fptree $(PWD)/testcases/5/test5.dat 98.0 $(PWD)/testcases/5/my_out.dat
	diff -daZ testcases/5/out_98.dat_sorted testcases/5/my_out.dat

testap: $(APP_DIR)/$(TARGET)

	./$(APP_DIR)/$(TARGET) apriori $(PWD)/testcases/1/test1.dat 30.0 $(PWD)/testcases/1/my_out.dat
	diff -daZ testcases/1/out_30.dat_sorted testcases/1/my_out.dat
	
	./$(APP_DIR)/$(TARGET) apriori $(PWD)/testcases/2/test2.dat 30.0 $(PWD)/testcases/2/my_out.dat
	diff -daZ testcases/2/out_30.dat_sorted testcases/2/my_out.dat

	./$(APP_DIR)/$(TARGET) apriori $(PWD)/testcases/3/test3.dat 01.0 $(PWD)/testcases/3/my_out.dat
	diff -daZ testcases/3/out_1.dat_sorted testcases/3/my_out.dat

	./$(APP_DIR)/$(TARGET) apriori $(PWD)/testcases/4/test4.dat 90.0 $(PWD)/testcases/4/my_out.dat
	diff -daZ testcases/4/out_90.dat_sorted testcases/4/my_out.dat

	./$(APP_DIR)/$(TARGET) apriori $(PWD)/testcases/5/test5.dat 98.0 $(PWD)/testcases/5/my_out.dat
	diff -daZ testcases/5/out_98.dat_sorted testcases/5/my_out.dat

