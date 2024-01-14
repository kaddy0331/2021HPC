#ifndef PARALLEL_FOR_H
#define PARALLEL_FOR_H

#include <pthread.h>

struct for_index {
    int start;
    int end;
    int increment;
    void *(*functor)(void *, int);
    void *arg;
};

void parallel_for(int start, int end, int increment, void *(*functor)(void *, int), void *arg, int num_threads);

#endif

