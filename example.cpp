#include <iostream>
#include <chrono>
#include <random>
#include "tbb/tbb.h"

#include "SSBTree.h"
using namespace std;
using namespace thu_ltl;
#define TEST_LAYOUT_NAME "thu_ltl"
static inline int file_exists(char const *file)
{
    return access(file, F_OK);
}
void run(char **argv)
{
    std::cout << "Simple Eample of SSBTree" << std::endl;
    int n = std::atoll(argv[1]);
    uint64_t *keys = new uint64_t[n];

    for (int i = 0 ; i < n; i++)
    {
        keys[i] = i + 1;
    }
    int num_thread = atoi(argv[2]);
    tbb::task_scheduler_init init(num_thread);
    SSBTree *KV =  nullptr;
    PMEMobjpool *pop;
    if (file_exists(argv[3]) != 0)
    {
        int sds_write_value = 0;
        pmemobj_ctl_set(NULL, "sds.at_create", &sds_write_value);
        if ((pop = pmemobj_create(argv[3], POBJ_LAYOUT_NAME(TEST_LAYOUT_NAME),
                                  8000000000, 0666)) == NULL)
        {
            printf("failed to create pool. path is : %s\n", argv[3]);
            delete[] keys;
            return ;
        }
        TOID(SSBTree) sbt = POBJ_ROOT(pop, SSBTree);
        KV = D_RW(sbt);
        KV->reStart(pop);
    }
    else
    {

        if ((pop = pmemobj_open(argv[3], POBJ_LAYOUT_NAME(TEST_LAYOUT_NAME))) == NULL)
        {
            printf("failed to open pool. path is : %s\n", argv[3]);
            delete[] keys;
            return ;
        }
        TOID(SSBTree) sbt = POBJ_ROOT(pop, SSBTree);
        KV = D_RW(sbt);
        KV->reStart(pop);
        delete[] keys;
        pmemobj_close(pop);
        return;
    }

    KV->pmdk_constructor(18, 36);
    {
        auto starttime = std::chrono::system_clock::now();
        tbb::parallel_for(tbb::blocked_range<uint64_t>(0, n), [&](const tbb::blocked_range<uint64_t> &range)
        {

            auto t = KV->getThreadInfo();
            for (uint64_t i = range.begin(); i != range.end(); i++)
            {
                KV->put(keys[i], keys[i], t);
            }
        });
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now() - starttime);

        printf("Throuoghput:put,%d,%f ops/us\n", n, (n * 1.0) / duration.count());
        printf("Elapsed time: put,%d,%f sec\n", n, duration.count() / 1000000.0);
    }

    {
        auto starttime = std::chrono::system_clock::now();

        tbb::parallel_for(tbb::blocked_range<uint64_t>(0, n), [&](const tbb::blocked_range<uint64_t> &range)
        {
            auto t = KV->getThreadInfo();
            for (uint64_t i = range.begin(); i != range.end(); i++)
            {
                //Keys[i] = Keys[i]->make_Key();
                uint64_t val = KV->lookup(keys[i], t);
                if (val != keys[i])
                {
                    std::cout << "get wrong value: " << val << "expected : " << keys[i] << std::endl;
                    throw;
                }
            }
        });
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::system_clock::now() - starttime);
        printf("Throuoghput:lookup,%d,%f ops/us\n", n, (n * 1.0) / duration.count());
        printf("Elapsed time: lookup,%d,%f sec\n", n, duration.count() / 1000000.0);
    }

    pmemobj_close(pop);
    delete[] keys;
}
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("usage: %s [n] [nthreads] poolpath \n n:number of keys (integer)\nnthreads:number of threads (integer)\npoolpath:<file-name>\n", argv[0]);
        return 1;
    }
    run (argv);
    return 0;
}