#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define ARRAY_SIZE 1000

int a[ARRAY_SIZE];
int global_index = 0;
int sum = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // 互斥锁声明

// 用于线程的求和函数
void* threadSum(void* arg) {
    int local_sum = 0;
    
    while (1) {
        int index;
        
        // 加锁以获取下一个未加元素的索引
        pthread_mutex_lock(&mutex);
        index = global_index;
        global_index++;
        pthread_mutex_unlock(&mutex);
        
        if (index < ARRAY_SIZE) {
            local_sum += a[index];
        } else {
            break; // 所有元素已经被处理
        }
    }
    
    // 加锁以更新全局求和
    pthread_mutex_lock(&mutex);
    sum += local_sum;
    pthread_mutex_unlock(&mutex);
    
    pthread_exit(NULL);
}

int main(int argc, char * argv[]) {

    int NUM_THREADS = atoi(argv[1]); // 选择线程数量

    pthread_t threads[NUM_THREADS];
    struct timespec start, end;
    
    // 初始化数组a
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i + 1; // 填充数组a
    }
    
    clock_gettime(CLOCK_MONOTONIC, &start); // 记录开始时间
    
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
    
    clock_gettime(CLOCK_MONOTONIC, &end); // 记录结束时间
    
    // 计算执行时间（以毫秒为单位）
    double start_time = (double)start.tv_sec * 1e3 + (double)start.tv_nsec * 1e-6;
    double end_time = (double)end.tv_sec * 1e3 + (double)end.tv_nsec * 1e-6;
    double execution_time = end_time - start_time;
    
    // 打印总和和执行时间
    printf("Sum: %d\n", sum);
    printf("Execution time: %f milliseconds\n", execution_time);
    
    return 0;
}

