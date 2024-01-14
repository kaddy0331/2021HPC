#include <stdio.h>
#include <cstdlib>
#include <pthread.h>
#include "parallel_for.h"

void *worker(void *args) {
    struct for_index *index = (struct for_index *)args;
    for (int i = index->start; i < index->end; i += index->increment) {
        index->functor(index->arg, i);
    }
    return NULL;
}

void parallel_for(int start, int end, int increment, void *(*functor)(void *, int), void *arg, int num_threads) {
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    struct for_index *indices = (struct for_index *)malloc(num_threads * sizeof(struct for_index));

    int chunk_size = (end - start + num_threads - 1) / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        indices[i].start = i * chunk_size;
        indices[i].end = (i + 1) * chunk_size;
        indices[i].increment = increment;
        indices[i].functor = functor;
        indices[i].arg = arg;

        // 创建线程并传递相应的结构体
        pthread_create(&threads[i], NULL, worker, (void *)&indices[i]);
    }

    // 等待线程完成
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], NULL);
    }

}

