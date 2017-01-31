#include <iostream>
#include <thread>
#include <mutex> 
#include <atomic>
#include <condition_variable>
#include <functional>
#include <Eigen/Dense>
#include <matrixreader.h>

using namespace Eigen;


/**
 * Механизм для синхронизации произвольного количества потоков (max_count),
 * позволяющий потокам останавливаться в заданной точке и возобновлять
 * работу только когда каждый из потоков достиг этой точки. Приостановленные потоки
 * возобновляются автоматически посредством оповещения.
 */
class Barrier {

    std::mutex mutex;
    std::condition_variable cv;

    volatile int count;
    volatile int max_count;
    volatile int generation;

public:

    Barrier(int max_count) :
        count (max_count),
        max_count (max_count),
        generation (0)
    {}
    
    void syncronize() {
        
        std::unique_lock<std::mutex> lock(mutex);
        
        int gen = generation;
        
        if (--count == 0) {
            generation++;
            count = max_count;
            cv.notify_all();
        }
        else {
            cv.wait(
                lock,
                [&] { return gen != generation; });
        }
    }
};


/*
 * Класс, осуществляющий распараллеливание заданной процедуры на непересекающемся разбиении данных,
 * расчитанный на многократный перезапуск задачи (треды создаются один раз).
 */
class ParallelExecutor {

protected:

    std::vector<std::thread> threads;
    std::atomic<bool>        is_terminated;
    Barrier barrier;

public:

    ParallelExecutor(int threads_num, std::function<void(int)> kernel) :
        is_terminated(false),
        barrier(threads_num + 1) // Барьер создается с учетом внешнего треда, использующего этот класс
    {
        for (int thread = 0; thread < threads_num; thread++) {

            threads.push_back(std::thread([&, thread, kernel] {

                while (true) {

                    barrier.syncronize();

                    if (is_terminated.load()) {
                       return;
                    }

                    // В вызываемую процедуру передается номер треда для осуществления
                    // партициирования данных в её реализации
                    
                    kernel(thread);

                    barrier.syncronize();
                }
            }));
        }
    }

    /*
     * Функция для вызова из внешнего управляющего треда, блокирует его до
     * тех пор пока каждый из тредов не выполнит свою часть работы.
     */
    void runKernels() {
        
        barrier.syncronize();
        barrier.syncronize();
    }

    void terminate() {
        
        is_terminated.store(true);

        barrier.syncronize();

        for (auto& t : threads) {
            t.join();
        }
    }

    virtual ~ParallelExecutor() {
        terminate();
    }
};


/**
 * Класс для поиска собственных чисел квадратной матрицы с помощью QR-алгоритма.
 * QR-разложение произовдится с помощью поворотов Гивенса в несколько потоков.
 */
template<typename MatrixType>
class EigenvalueSolver {

protected:

    using Scalar         = typename MatrixType::Scalar;
    using RealScalar     = typename NumTraits<Scalar>::Real;
    using RowMajorMatrix = Matrix<Scalar, Dynamic, Dynamic, RowMajor>; 
    using Vector         = Matrix<Scalar, Dynamic, 1>;

    /*
        По умолчанию в Eigen используется колоночный формат хранения.
        Использование RowMajor-формата в данной задаче дает значительный прирост скорости (~30%)
    */
    RowMajorMatrix matrix;      
    Vector         eigenvalues;
    
    int n;
    int threads_num;
    int max_iterations;

    // Временные данные для расчета 
    
    MatrixType   Q;
    volatile int current_band;

public:
    
    EigenvalueSolver(
        const MatrixType& matrix,
        int threads_num     = std::thread::hardware_concurrency(),
        int max_iterations  = 1000
    ) :
        matrix (matrix),
        threads_num (threads_num),
        max_iterations (max_iterations)
    {
        if (matrix.rows() != matrix.cols()) {
            throw std::invalid_argument("Matrix must be square");
        }
        
        if (matrix.rows() > 10000) {
            throw std::invalid_argument("Matrix must be not larger than 10000x10000");
        }

        n = matrix.rows();

        std::cout << "Computing eigenvalues for " << n << "x" << n << " matrix" << std::endl;
        std::cout << "threads_num    = " << threads_num     << std::endl;
        std::cout << "max_iterations = " << max_iterations  << std::endl;

        compute();
    }
   
    const Vector& getEigenvalues() {
        return eigenvalues;
    }

protected:


    void compute() {
       
        // Подготавливаем треды 
       
        ParallelExecutor executor(
            threads_num,
            std::bind(
                &EigenvalueSolver::bandRotationsKernel,
                this,
                std::placeholders::_1));


        // Проделываем итерации QR-алгоритма

        for (int i = 0; i < max_iterations; i++) {
           
            // Подготавливаем новую единичную матрицу Q.
            // В ней мы будем запоминать повороты Гивенса, которые применялись к исходной матрице слева,
            // чтобы потом тоже самое применить справа.
            // (...G2*G1) * M = Q^t * M

            Q.setIdentity(n, n);
           
            /*
                Распараллеливаем задачу приведения исходной матрицы к верхней треугольной форме.
                Один поворот Гивенса обнуляет ечейку (i, j) модифицируя строчки i и i-1,
                в идеале за раз можно параллельно осуществить n/2 поворота.
                
                То же самое автоматически следует для матрицы Q, только она будет модифицировать колонки.
                Чтобы обнуленные ячейки снова не модифицировались, нужно чтобы при повороте они шли парами,
                одна над другой.
                
                Представим матрицу в виде осей координат (x, y), c началом в нижнем левом углу.
                Лего заметить, что условие будет выполняться если обнулять ячейки, лежащие на наклонной 
                прямой y = -2*x + b, последовательно сдвигая её от точки (0, 0) в сторону главной диагонали,
                последовательно увеличивая b.
                
                Для простоты положим: m = n - 1
                Главная диагональ:    y = -x + m
                Тогда точки пересечения нашей прямой с осью Y, главной диагональю и осью X соответственно

                A = (0, b)
                B = (b - m, 2*m - b)
                C = (b / 2, 0) 
                
                Отсюда получаем нужные нам промежутки:

                b = [0 : 2 * (m - 1)]
                x = (B_x : C_x] = (b - m : b / 2]
                y = (B_y : C_y] = (2*m - b : 0]

                j = x = [b - m + 1 : b / 2]  ( +1 т.к. нам не нужен элемент на главной диагонале) 
                i = m - y = m + 2*j - b

                В результате получаем параметры для циклов и разбиений:

                b = 0 : 2 * (n - 2)
                j = [b - n + 2 : b / 2] 
                i = n + 2*j - b - 1 
            */

            
            for (int b = 0; b <= 2 * (n - 2); b++) {
               
                // Запускаем задачу для одной наклонной линии, значения на которой будут обнуляться
                
                current_band = b; 

                executor.runKernels();
            }
            
            // 3. Матрица Q была заполнена, применяем преобразование к правой части исходной матрицы 

            matrix *= Q;
            
            // 4. TODO:
            // 
            // 1) Проверять сходимость 
            // 2) Проверять вторую диагональ
            // 2) Добавить сдвиги
           
            //std::cout << "Iteration " << i << std::endl;
        }

        // 3. Получаем результат

        eigenvalues = matrix.diagonal();
    }
   

    /**
     * Процедура выполняется в некотором количестве тредов,
     * часть данных для обработки выбирается исходя из номера треда
     */
    void bandRotationsKernel(int thread) {

        int b = current_band;

        for (
            int j = std::max(0, b - n + 2) + thread;
            
            j <= b / 2;
            
            j += threads_num
        ) {
            int i = n + 2 * j - b - 1;

            applyGivensRotationForCell(i, j);
        }
    }


    /**
     * Один поворот Гивенса обнуляет ечейку (i, j) и модифицирует строки i-1 и i 
     */
    void applyGivensRotationForCell(int i, int j) {
        
        JacobiRotation<Scalar> G;

        G.makeGivens(
            matrix(i - 1, j),
            matrix(i, j));

        // Применяем поворот

        matrix.applyOnTheLeft(i - 1, i, G.adjoint());

        // Запоминаем поворот

        Q.applyOnTheRight(i - 1, i, G);
    }
};


int main()
{
    const int n = 1000;

    using MatrixType = MatrixXf;

    MatrixType A = MatrixType::Random(n, n);
    
    EigenvalueSolver<MatrixType> solver(A, 6, 100);
    std::cout << solver.getEigenvalues() << '\n' << std::endl;

    //MatrixXd B = matrixreader::readMatrix("mat.txt");
    //std::cout << A.eigenvalues() << '\n'  << std::endl;
    //HessenbergDecomposition<MatrixXd> hd(A);
    //std::cout << hd.matrixH().rows() << std::endl;
}

