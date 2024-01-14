#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

// 定义二次方程的系数
double a, b, c;
double x1, x2;
int solutions_found = 0;

// 定义互斥锁和条件变量
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

// 定义线程的返回值结构
typedef struct {
    double b_squared;
    double four_a_c;
} ThreadResult;

// 用于线程的求解函数，计算b²的部分
void* calculateBSquared(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    result->b_squared = b * b;
    pthread_exit(NULL);
}

// 用于线程的求解函数，计算4ac的部分
void* calculateFourAC(void* arg) {
    ThreadResult* result = (ThreadResult*)arg;
    result->four_a_c = 4 * a * c;
    pthread_exit(NULL);
}

int main() {
    pthread_t thread_b_squared, thread_four_a_c;
    ThreadResult result_b_squared, result_four_a_c;

    // 输入二次方程的系数
    printf("Enter the coefficients of the quadratic equation (a, b, c): ");
    scanf("%lf %lf %lf", &a, &b, &c);

    // 创建并启动两个线程，分别计算b²和4ac的部分
    pthread_create(&thread_b_squared, NULL, calculateBSquared, &result_b_squared);
    pthread_create(&thread_four_a_c, NULL, calculateFourAC, &result_four_a_c);

    // 等待两个线程结束
    pthread_join(thread_b_squared, NULL);
    pthread_join(thread_four_a_c, NULL);

    // 计算根
    double discriminant = result_b_squared.b_squared - result_four_a_c.four_a_c;

    if (a == 0) {
        if (b != 0) {
            x1 = x2 = -c / b;
            solutions_found = 1;
        } else if (c == 0) {
            solutions_found = -1; // 无穷多解
        }
    } else {
        if (discriminant > 0) {
            x1 = (-b + sqrt(discriminant)) / (2 * a);
            x2 = (-b - sqrt(discriminant)) / (2 * a);
            solutions_found = 2;
        } else if (discriminant == 0) {
            x1 = x2 = -b / (2 * a);
            solutions_found = 1;
        }
    }

    // 输出结果
    if (solutions_found == -1) {
        printf("Infinite solutions\n");
    } else if (solutions_found == 0) {
        printf("No real solutions\n");
    } else if (solutions_found == 1) {
        printf("One real solution: x = %lf\n", x1);
    } else {
        printf("Two real solutions: x1 = %lf, x2 = %lf\n", x1, x2);
    }

    return 0;
}

