// Copyright for SSBTree is held by the Tsinghua University
// Licensed under the MIT license.
// Authors:
// Tongliang Li <onceltl@gmail.com>

#ifndef SSBTREE_H
#define SSBTREE_H

#include <atomic>
#include <stdint.h>
#include <libpmemobj.h>
#include <cmath>
#include "Epoche.h"
namespace thu_ltl
{
    class Node;
    class SSBTree;
    POBJ_LAYOUT_BEGIN(thu_ltl);
    POBJ_LAYOUT_ROOT(thu_ltl, SSBTree);
    POBJ_LAYOUT_TOID(thu_ltl, Node);
    POBJ_LAYOUT_END(thu_ltl);
    typedef uint64_t Oidoff;
    struct Pair //16bytes
    {
        uint64_t key;
        Oidoff value;
    };

    static  constexpr uint32_t maxPairsLength = 35;
    static  constexpr uint32_t NodeSize = 1280;
    static  constexpr uint32_t highPosition = 50;
    static  constexpr int midindex = 23;
    static_assert((maxPairsLength + 5) * 32 == NodeSize, "NodeSize should match maxPairsLength");
    static_assert(highPosition >= 48, "cacnonical addreses at least 48 bit.");
    static_assert(exp2(64 - highPosition) > 15, "length should smaller than Noncanonical addresses ");


    class  Node
    {
    public:

        /****header_layout****/
        // version(16bit) number(16bit)
        // Lazyboxflag(2bit) bottomflag(1bit) Obsolete(1bit)
        // Right(2bit) mutex(2bit)
        // reserve(24bit)
        /********************/
        volatile uint64_t header;//8Byte
        uint64_t dummy;  //8Byte
        Pair LazyBox;           //16Bytes
        uint64_t midkey[2]; // 16Bytes
        volatile uint64_t maxKey[2];   //16Bytes
        volatile Oidoff right[2];       //16bytes
        Pair pairs[2 * maxPairsLength];
        uint64_t dummy2[2];
        PMEMmutex mutex;   // 64 bytes
    public:
        static inline void addRight(uint64_t &header) __attribute__((always_inline));
        static inline bool ReadcheckVesion(uint64_t ol, uint64_t ne) __attribute__((always_inline));
        static inline bool WritecheckVesion(uint64_t ol, uint64_t ne) __attribute__((always_inline));
        static inline bool RightCheck(uint64_t ol, uint64_t ne) __attribute__((always_inline));
        static inline void clflush(PMEMobjpool *pop, char *data, int len, bool front, bool back) __attribute__((always_inline));
    };

    class SSBTree
    {
    private:
        TOID(Node) headoid, tailoid, rootoid;

        //Trigger a merge when the total num of keys in consecutive siblings (no parent) < Lnum.
        //Trigger a split when the total num of keys in a node > Rnum.
        //Lnum & Rnum are only useful if REBALANCE is set.
        uint32_t Lnum, Rnum;
        uint64_t epocheColor;
        Epoche *epoche;
        PMEMobjpool *pop;
    private:

        int64_t signextend(const uint64_t x);

        Node *newNode();
        void linear_search(int &k,  Pair *offset_pair, const int &n, const uint64_t &findkey);
        void split(Node *node);
        void merge(Node *node, ThreadInfo &threadEpocheInfo);
        //insert a k-v pair into a node
        void upKey(Node *&node, uint64_t &header, int &lazyflag, Pair &lazybox, Pair &upPair,
                   Pair *&move_pair, Pair *&offset_pair,
                   int &LessOrEqual, int &endlocation);

        //delete a k-v pair from a node
        void downKey(Node *&node, uint64_t &header, int &lazyflag, Pair &lazybox, Pair &downPair,
                     Pair *&move_pair, Pair *&offset_pair,
                     int &LessOrEqual, int &endlocation, ThreadInfo &threadEpocheInfo);
        void leafscan(TOID(Node) nodeoid, const uint64_t minscan, const uint64_t maxscan, int length, uint64_t *results, int &offset);

    public:

        SSBTree(const SSBTree &) = delete;
        SSBTree(uint32_t lnum, uint32_t rnum);
        ~SSBTree();

        void pmdk_constructor(uint32_t lnum, uint32_t rnum);
        void reStart(PMEMobjpool *setpop);
        ThreadInfo getThreadInfo();

        uint64_t lookup(const uint64_t findkey, ThreadInfo &threadEpocheInfo);
        void update(const uint64_t updatekey, const uint64_t updatevalue, ThreadInfo &threadEpocheInfo);
        void normalRemove(const uint64_t removekey, ThreadInfo &threadEpocheInfo);
        void balanceRemove(const uint64_t removekey, ThreadInfo &threadEpocheInfo);
        void remove(const uint64_t removekey, ThreadInfo &threadEpocheInfo);
        void put(const uint64_t insertKey, const uint64_t insertValue, ThreadInfo &threadEpocheInfo);
        void scan(const uint64_t minscan, const uint64_t maxscan, int length, uint64_t *results, int &offset, ThreadInfo &threadEpocheInfo);
    };
}


#endif //SSBTREE_H
