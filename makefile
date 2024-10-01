.PHONY: build
build: clean
	python sgrr/sgcc/lhrr/setup.py build_ext --inplace

clean: 
	rm -f -r build/
	rm -f sgrr/sgcc/lhrr/lhrr.cpp
