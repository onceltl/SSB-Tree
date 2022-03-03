#include "SSBTree_pibench_wrapper.h"

#define TEST_LAYOUT_NAME "thu_ltl"
using namespace thu_ltl;
extern "C" tree_api *create_tree(const tree_options_t &opt)
{
    return new ssbtree_wrapper(opt);
}

static inline int file_exists(char const *file)
{
    return access(file, F_OK);
}
inline int64_t signextend(const uint64_t x)
{
    struct
    {
        int64_t x: 48;
    } s;
    return s.x = x;
}

SSBTree *create_new_tree_in(const tree_options_t &opt)
{
    SSBTree *KV =  nullptr;
    PMEMobjpool *pop;
    int sds_write_value = 0;
    pmemobj_ctl_set(NULL, "sds.at_create", &sds_write_value);
    if ((pop = pmemobj_create(opt.pool_path.c_str(), POBJ_LAYOUT_NAME(TEST_LAYOUT_NAME), opt.pool_size, 0666)) == NULL)
    {
        printf("failed to create pool. path is : %s\n", opt.pool_path.c_str());
        return nullptr;
    }
    TOID(SSBTree) sbt = POBJ_ROOT(pop, SSBTree);
    KV = D_RW(sbt);
    KV->reStart(pop);
    KV->pmdk_constructor(14, 27);
    printf("open pool successfully %s\n", opt.pool_path.c_str());
    return KV;
}

SSBTree *recovery_from_pool(const tree_options_t &opt)
{
    SSBTree *KV =  nullptr;
    PMEMobjpool *pop;
    if ((pop = pmemobj_open(opt.pool_path.c_str(), POBJ_LAYOUT_NAME(TEST_LAYOUT_NAME))) == NULL)
    {
        printf("failed to open pool. path is : %s\n", opt.pool_path.c_str());
        return nullptr;
    }
    TOID(SSBTree) sbt = POBJ_ROOT(pop, SSBTree);
    KV = D_RW(sbt);
    KV->reStart(pop);
    printf("open pool successfully %s\n", opt.pool_path.c_str());
    pmemobj_close(pop);
    return KV;
}

ssbtree_wrapper::ssbtree_wrapper(const tree_options_t &opt)
{
    if (file_exists(opt.pool_path.c_str()) != 0)
    {

        printf("creating new tree on pool.");
        tree_ = create_new_tree_in(opt);
    }
    else
    {
        printf("recovery from existing pool.");
        tree_ = recovery_from_pool(opt);
    }
}

ssbtree_wrapper::~ssbtree_wrapper()
{
}

bool ssbtree_wrapper::find(const char *key, size_t key_sz, char *value_out)
{
    // FIXME(tzwang): for now only support 8-byte values
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    auto t = tree_->getThreadInfo();
    uint64_t ans = tree_-> lookup(k, t);
    memcpy(value_out, &ans, sizeof(uint64_t));
    return 1;
}

bool ssbtree_wrapper::insert(const char *key, size_t key_sz, const char *value,
                             size_t value_sz)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    auto t = tree_->getThreadInfo();
    uint64_t v = *reinterpret_cast<uint64_t *>(const_cast<char *>(value));
    tree_->put(k, signextend(v), t);
    return 1;
}

bool ssbtree_wrapper::update(const char *key, size_t key_sz, const char *value,
                             size_t value_sz)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    auto t = tree_->getThreadInfo();
    uint64_t v = *reinterpret_cast<uint64_t *>(const_cast<char *>(value));

    tree_->update(k, signextend(v), t);

    return 1;
}

bool ssbtree_wrapper::remove(const char *key, size_t key_sz)
{
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    auto t = tree_->getThreadInfo();
    tree_->remove(k, t);
    return 1;
}

int ssbtree_wrapper::scan(const char *key, size_t key_sz, int scan_sz,
                          char *&values_out)
{
    static thread_local std::array < uint64_t, 100> results;
    uint64_t k = *reinterpret_cast<uint64_t *>(const_cast<char *>(key));
    auto t = tree_->getThreadInfo();
    int resultsFound = 0;
    tree_->scan(k, UINT64_MAX, scan_sz, results.data(), resultsFound, t);
    return resultsFound;
}