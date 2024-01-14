#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define NUM_POINTS 1000000

int NUM_THREADS;
double area = 0.0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

// 用于线程的估算函数
void* estimateArea(void* arg) {
    int points_per_thread = NUM_POINTS / NUM_THREADS;
    int points_inside = 0;

    for (int i = 0; i < points_per_thread; i++) {
        double x = (double)rand() / RAND_MAX; // 随机生成 x 值
        double y = (double)rand() / RAND_MAX; // 随机生成 y 值

        if (y <= x * x) {
            points_inside++;
        }
    }

    pthread_mutex_lock(&mutex);
    area += (double)points_inside / points_per_thread;
    pthread_mutex_unlock(&mutex);

    pthread_exit(NULL);
}

int main(int argc, char * argv[]) {
    NUM_THREADS = atoi(argv[1]); // 选择线程数量
    srand(time(NULL));
    pthread_t threads[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_create(&threads[i], NULL, estimateArea, NULL) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("pthread_join");
            return 1;
        }
    }

    area /= NUM_THREADS; // 计算平均值

    printf("Estimated area: %f\n", area);
    
    return 0;
}

