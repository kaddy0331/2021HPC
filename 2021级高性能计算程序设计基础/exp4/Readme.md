fft_serial：  
g++ -shared -fPIC -o libparallel_for.so parallel_for.cpp lpthread  
Export LD_LIBRARY_PATH=/path/to/library:$LD_LIBRARY_PATH  
g++ -o fft_serial fft_serial.cpp -I/path/to/parallel -L. -lparallel_for -lpthread -lm  
./fft_serial 线程数  
  
heated_plate_openmp:  
mpic++ -o heated_plate_openmp heated_plate_openmp.cpp -lm  
mpirun -n 线程数 ./heated_plate_openmp  
