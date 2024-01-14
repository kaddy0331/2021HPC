#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000
#define NUM_ELEMENTS_PER_GROUP 10

int NUM_GROUPS = ARRAY_SIZE / NUM_ELEMENTS_PER_GROUP;

int a[ARRAY_SIZE];
int group_sums[ARRAY_SIZE / NUM_ELEMENTS_PER_GROUP];
int global_group = 0; // 修正全局变量名
int sum = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 用于线程的求和函数
void* threadSum(void* arg) {
    int local_group_sum = 0;
    
    while (1) {
        int group;
        
        // 加锁以获取下一个组的索引
        pthread_mutex_lock(&mutex);
        group = global_group;
        global_group++;
        pthread_mutex_unlock(&mutex);
        
        if (group < NUM_GROUPS) {
            int group_start = group * NUM_ELEMENTS_PER_GROUP;
            int group_end = group_start + NUM_ELEMENTS_PER_GROUP;
            for (int i = group_start; i < group_end; i++) {
                local_group_sum += a[i];
            }
        } else {
            break; // 所有组已经被处理
        }
    }
    
    // 加锁以更新总和
    pthread_mutex_lock(&mutex);
    sum += local_group_sum;
    pthread_mutex_unlock(&mutex);
    
    pthread_exit(NULL);
}

int main(int argc, char *argv[]) {

    int NUM_THREADS = atoi(argv[1]);

    pthread_t threads[NUM_THREADS];
    struct timespec start, end;
    
    // 初始化数组a
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i + 1;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // 创建和启动线程
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, threadSum, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }
    
    // 等待线程结束
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return 1;
        }
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double start_time = (double)start.tv_sec * 1e3 + (double)start.tv_nsec * 1e-6;
    double end_time = (double)end.tv_sec * 1e3 + (double)end.tv_nsec * 1e-6;
    double execution_time = end_time - start_time;
    
    printf("Sum: %d\n", sum);
    printf("Execution time: %f milliseconds\n", execution_time);
    
    return 0;
}

