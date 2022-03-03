// Copyright for SSBTree is held by the Tsinghua University
// Licensed under the MIT license.
// Authors:
// Tongliang Li <onceltl@gmail.com>
#include <algorithm>
#include <assert.h>
#include <cstring>
#include <new>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <emmintrin.h>
#include <immintrin.h>
#include <libpmemobj.h>
#include "SSBTree.h"
#include "Epoche.cpp"
namespace thu_ltl
{

    /****header_layout****/
    // version(16bit) number(16bit)
    // Lazyboxflag(2bit) bottomflag(1bit) Obsolete(1bit)
    // Right(2bit) mutex(2bit)
    // reserve(24bit)
    /********************/
#define VERSION_BITS (0xFFFFULL << 48)
#define NUM_BITS (0xFFFFULL << 32)
#define BOX_BITS (3ULL << 30)
#define BOTTOM_BITS (1ULL <<29)
#define DEL_BITS (1ULL<<28)
#define RIGHT_BITS (3ULL<<26)
#define LOCK_BITS (3ULL<<24)
#define isObsolete(x) ((x&DEL_BITS)!=0)
#define isBottom(x) ((x&BOTTOM_BITS)!=0)
#define rightTurn(x) ((x>>26)&1)
#define versionTurn(x) ((x>>48)&1)
#define addbox_BITS (1ULL << 30)
#define delbox_BITS (1ULL << 31)
#define addVersion_BITS (1ULL << 48)
#define addNum_BITS (1ULL << 32)
#define addRight_BITS (1ULL << 26)
#define getNum(x) ((x >> 32)&0xFFFF)

    static constexpr int shifversion = 48;
    static constexpr int shifnumber = 32;
    static constexpr int shiflazybox = 30;
    static constexpr int shifmutex = 24;
    static constexpr int lazydiff[4] = {0, 1, -1, 0};

    static constexpr unsigned long cache_line_size = 64;
    static inline void prefetch_(const void *ptr)
    {
        typedef struct
        {
            char x[cache_line_size];
        } cacheline_t;
        asm volatile("prefetcht0 %0" : : "m" (*(const cacheline_t *)ptr));
    }
    inline void Node::addRight(uint64_t &header)
    {
        uint64_t right = header & RIGHT_BITS;
        header ^= right;
        right = (right + addRight_BITS) &RIGHT_BITS;
        header |= right;
    }

    inline void Node::clflush(PMEMobjpool *pop, char *data, int len, bool front, bool back)
    {
        volatile char *ptr = (char *)((unsigned long)data & ~(cache_line_size - 1));

        if (front)
            pmemobj_drain(pop);

        for (; ptr < data + len; ptr += cache_line_size)
        {

#ifdef CLFLUSH
            asm volatile("clflush %0" : "+m" (*(volatile char *)ptr));
#elif CLFLUSH_OPT
            asm volatile(".byte 0x66; clflush %0" : "+m" (*(volatile char *)(ptr)));
#elif CLWB
            asm volatile(".byte 0x66; xsaveopt %0" : "+m" (*(volatile char *)(ptr)));
#endif
        }
        if (back)
            pmemobj_drain(pop);

    }
    //Optimized Optimistic Concurrency Control
    inline bool Node::ReadcheckVesion(uint64_t ol, uint64_t ne)
    {
        return
            (ol >> shifversion == ne >> shifversion) //oldversion = newversion
            || ( (ol >> shifversion) + 1 == (ne >> shifversion)  && ((ne >> shiflazybox) & 3) == 0  );
        //  || (oldversion + 1 ==  newversion        &&              flag == 0 )
    }
    inline bool Node::RightCheck(uint64_t ol, uint64_t ne )
    {
        return (ol & RIGHT_BITS) == (ne & RIGHT_BITS);
    }
    inline bool Node::WritecheckVesion(uint64_t ol, uint64_t ne)
    {
        return (ol >> shifnumber) == (ne >> shifnumber);
    }

    ThreadInfo SSBTree::getThreadInfo()
    {
        return ThreadInfo(*(this->epoche));
    }

    void SSBTree::reStart(PMEMobjpool *setpop)
    {
        pop = setpop;
        epoche = new Epoche(256);
        pobj_alloc_class_desc AllocClass;
        AllocClass.unit_size = NodeSize;
        AllocClass.alignment = 256; // physical granularity of DCPMM
        AllocClass.units_per_block = 1;
        AllocClass.class_id = 128;
        AllocClass.header_type = POBJ_HEADER_NONE;

        if (pmemobj_ctl_set(pop, "heap.alloc_class.128.desc", &AllocClass))
        {
            printf("alloc_clas failed. \n");
            pmemobj_close(pop);
            exit(-1);
        }
    }

    void SSBTree::pmdk_constructor(uint32_t lnum, uint32_t rnum)
    {
        Lnum = lnum;
        Rnum = rnum;
        epoche = new Epoche(256);
        uint64_t header = BOTTOM_BITS + addNum_BITS;
        pmemobj_xalloc(pop, &tailoid.oid, sizeof(Node), 0, POBJ_CLASS_ID(128), NULL, NULL);
        Node *tail = D_RW(tailoid);
        tail->header = (header);
        tail->pairs[0].key = -1;
        tail->pairs[0].value = 0;
        Node::clflush(pop, (char *) tail, sizeof(Node), false, true);


        pmemobj_xalloc(pop, &headoid.oid, sizeof(Node), 0, POBJ_CLASS_ID(128), NULL, NULL);
        Node *head = D_RW(headoid);
        head->header = (header);
        head->right[0] = tailoid.oid.off;
        head->maxKey[0] = -1;
        head->pairs[0].key = 0;
        head->pairs[0].value = 0;
        Node::clflush(pop, (char *) head, sizeof(Node), false, true);

        header = addNum_BITS;

        TOID(Node) typeNode;
        pmemobj_xalloc(pop, &typeNode.oid, sizeof(Node), 0, POBJ_CLASS_ID(128), NULL, NULL);
        Node *newhead = D_RW(typeNode);
        newhead->header = (header);
        newhead->right[0] = tailoid.oid.off;
        newhead->maxKey[0] = -1;
        newhead->pairs[0].key = 0;
        newhead->pairs[0].value = headoid.oid.off;
        Node::clflush(pop, (char *)newhead, sizeof(Node), false, true);
        rootoid = headoid;
        headoid = typeNode;
        Node::clflush(pop, (char *)this, sizeof(SSBTree), false, true);
    }
    SSBTree::SSBTree(uint32_t lnum = maxPairsLength, uint32_t rnum = maxPairsLength * 8): Lnum(lnum), Rnum(rnum)
    {
    }


    SSBTree::~SSBTree()
    {
    }

    inline int64_t SSBTree::signextend(const uint64_t x)
    {
        struct
        {
int64_t x:
            highPosition;
        } s;
        return s.x = x;
    }

    void SSBTree::upKey(Node *&node, uint64_t &header, int &lazyflag, Pair &lazybox, Pair &upPair,
                        Pair *&move_pair, Pair *&offset_pair,
                        int &LessOrEqual, int &endlocation)

    {
        uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
        int w1 = (v ^ signextend(v)) >> highPosition;
        int w2 = LessOrEqual + 1;
        if (lazyflag == 0x2)
        {
            if (w1 < w2) w2--;
        }

        if (lazyflag == 1)
        {

            if (w1 < w2) w2++;
            else if (w1 > w2) w1++;
            else if (lazybox.key > upPair.key) w1++;
            else w2++;
        }
        if (lazyflag == 1) //two inserts -> COW
        {
            if (w2 > w1 && w2 - 1 > endlocation) //append
            {
                offset_pair[w2 - 1] = upPair;
                Node::clflush(pop, (char *)&offset_pair[w2 - 1], sizeof(Pair), false, false);
                if (w2 - 1 == midindex)
                    node->midkey[versionTurn(header)] = upPair.key;
                node->header = (header + addNum_BITS + 2 * addVersion_BITS);
                Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);
            }
            else
            {

                move_pair[w1].key = lazybox.key;
                move_pair[w1].value = signextend(lazybox.value);
                move_pair[w2] = upPair;
                if (w1 > w2 ) std::swap(w1, w2);
                memcpy(move_pair, offset_pair, w1 * sizeof(Pair)); //0~w1-1
                memcpy(move_pair + w1 + 1, offset_pair + w1, (w2 - w1 - 1)*sizeof(Pair)); //w1+1~w2-1
                memcpy(move_pair + w2 + 1, offset_pair + w2 - 1, (endlocation - w2 + 2)*sizeof(Pair)); //w2+1~end
                Node::clflush(pop, (char *)move_pair, (endlocation + 3) * sizeof(Pair), false, false);
                if (endlocation + 2 >= midindex)
                    node->midkey[versionTurn(header) ^ 1] = move_pair[midindex].key;
                node->header = ((header ^ addbox_BITS) + addVersion_BITS + addNum_BITS);
                Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);

            }
        }
        else if (lazyflag == 0x2)   //one delete&one insert ->COW
        {
            move_pair[w2] = upPair;
            if (w1 <= w2)
            {
                memcpy(move_pair, offset_pair, w1 * sizeof(Pair)); //0~w1-1
                memcpy(move_pair + w1, offset_pair + w1 + 1, (w2 - w1)*sizeof(Pair)); //w1-1~w2-1
                memcpy(move_pair + w2 + 1, offset_pair + w2 + 1, (endlocation - w2)*sizeof(Pair)); //w2+1~end(ebd=oldend-1)
            }
            else
            {
                memcpy(move_pair, offset_pair, w2 * sizeof(Pair)); //0~w2-1
                memcpy(move_pair + w2 + 1, offset_pair + w2, (w1 - w2)*sizeof(Pair)); //w2~w1-1
                memcpy(move_pair + w1 + 1, offset_pair + w1 + 1, (endlocation - w1)*sizeof(Pair)); //w1+1~end
            }
            Node::clflush(pop, (char *)move_pair, (endlocation + 1) * sizeof(Pair), false, false);
            if (endlocation >= midindex)
                node->midkey[versionTurn(header) ^ 1] = move_pair[midindex].key;
            node->header = ((header ^ delbox_BITS) + addVersion_BITS + addNum_BITS);
            Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);
        }
        else
        {
            // insertKey if max
            if (w2 > endlocation) //append
            {
                offset_pair[w2] = upPair;
                Node::clflush(pop, (char *)&offset_pair[w2], sizeof(Pair), false, false);
                if (w2 == midindex)
                    node->midkey[versionTurn(header)] = upPair.key;
                node->header = (header + addNum_BITS + 2 * addVersion_BITS);
                Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);
            }
            else
            {
                node->LazyBox.key = upPair.key;
                node->LazyBox.value = upPair.value ^ ((uint64_t)w2 << highPosition);
                node->header = ((header | addbox_BITS) + addNum_BITS);
                Node::clflush(pop, (char *) &node->header, cache_line_size, false, true);

            }
        }

        split(node);
        if (node == D_RW(headoid))
        {
            TOID(Node) typeNode;
            pmemobj_xalloc(this->pop, &typeNode.oid, sizeof(Node), 0, POBJ_CLASS_ID(128), NULL, NULL);
            Node *newhead = D_RW(typeNode);


            pmemobj_mutex_zero(pop, &newhead->mutex);
            newhead->header = (addNum_BITS);
            newhead->right[0] = tailoid.oid.off;
            newhead->maxKey[0] = -1;
            newhead->pairs[0].key = 0;
            newhead->pairs[0].value = headoid.oid.off;
            Node::clflush(pop, (char *)newhead, cache_line_size + 2 * sizeof(Pair), false, true);
            rootoid = headoid;
            headoid = typeNode;
            Node::clflush(pop, (char *)this, sizeof(SSBTree), false, true);
        }
    }
    void SSBTree::downKey(Node *&node, uint64_t &header, int &lazyflag, Pair &lazybox, Pair &downPair,
                          Pair *&move_pair, Pair *&offset_pair,
                          int &LessOrEqual, int &endlocation, ThreadInfo &threadEpocheInfo)
    {
        uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
        int w1 = (v ^ signextend(v)) >> highPosition;
        int w2 = LessOrEqual;
        if (lazyflag == 1)
        {
            if (w2 < w1) w1--;
        }
        if (lazyflag == 2) //two de -> COW
        {
            if (downPair.key == lazybox.key)
                return;
            //if (w2==endlocation)
            //{
            //    node->header =(header-addNum_BITS);
            //    Node::clflush((char *) &node->header,sizeof(uint64_t),true,true);
            //} else
            {
                if (w1 > w2 ) std::swap(w1, w2);
                memcpy(move_pair, offset_pair, w1 * sizeof(Pair)); //0~w1-1
                memcpy(move_pair + w1 , offset_pair + w1 + 1, (w2 - w1 - 1)*sizeof(Pair)); //w1+1~w2-1
                memcpy(move_pair + w2 - 1, offset_pair + w2 + 1, (endlocation - w2)*sizeof(Pair)); //w2+1~end
                Node::clflush(pop, (char *)move_pair, (endlocation - 1)*sizeof(Pair), false, false);

                if (endlocation - 2 >= midindex)
                    node->midkey[versionTurn(header) ^ 1] = move_pair[midindex].key;

                node->header = ((header ^ delbox_BITS) + addVersion_BITS - addNum_BITS);
                Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);
            }
        }
        else if (lazyflag == 1)   //one delete&one insert ->COW
        {
            if (downPair.key == lazybox.key)
            {
                node->header = ((header ^ addbox_BITS) + addVersion_BITS + addVersion_BITS - addNum_BITS);
                Node *head = D_RW(headoid);
                TOID(Node) nheadoid = headoid;
                nheadoid.oid.off = head->pairs[0].value ;
                if (node == D_RW(nheadoid) && getNum(node->header) == 1 && !isBottom(node->header))
                {

                    epoche->markNodeForDeletion((void *)head, threadEpocheInfo);
                    head->header = (head->header | DEL_BITS);
                    Node::clflush(pop, (char *)&head->header, sizeof(uint64_t), false, true);
                    headoid = nheadoid;
                    rootoid.oid.off = D_RW(headoid)->pairs[0].value;
                    Node::clflush(pop, (char *)this, sizeof(SSBTree), false, true);
                }
                return;
            }
            std::swap(w1, w2);
            move_pair[w2].key = lazybox.key;
            move_pair[w2].value = signextend(reinterpret_cast<uint64_t>(lazybox.value));
            if (w1 <= w2)
            {
                memcpy(move_pair, offset_pair, w1 * sizeof(Pair)); //0~w1-1
                memcpy(move_pair + w1, offset_pair + w1 + 1, (w2 - w1)*sizeof(Pair)); //w1-1~w2-1
                memcpy(move_pair + w2 + 1, offset_pair + w2 + 1, (endlocation - w2)*sizeof(Pair)); //w2+1~end(ebd=oldend-1)
            }
            else
            {
                memcpy(move_pair, offset_pair, w2 * sizeof(Pair)); //0~w2-1
                memcpy(move_pair + w2 + 1, offset_pair + w2, (w1 - w2)*sizeof(Pair)); //w2~w1-1
                memcpy(move_pair + w1 + 1, offset_pair + w1 + 1, (endlocation - w1)*sizeof(Pair)); //w1+1~end
            }
            Node::clflush(pop, (char *)move_pair, (endlocation + 1) * sizeof(Pair), false, false);
            if (endlocation >= midindex)
                node->midkey[versionTurn(header) ^ 1] = move_pair[midindex].key;
            node->header = ((header ^ addbox_BITS) + addVersion_BITS - addNum_BITS);
            Node::clflush(pop, (char *) &node->header, sizeof(uint64_t), true, true);
        }
        else
        {

            //delete max key

            //if (w2 = endlocation)
            // {
            //    node->header =(header-addNum_BITS);
            //    Node::clflush((char *) &node->header,sizeof(uint64_t),true,true);
            //} else
            {
                node->LazyBox.key = downPair.key;
                node->LazyBox.value = ((uint64_t)w2 << highPosition);
                node->header = ((header | delbox_BITS) - addNum_BITS)  ;
                Node::clflush(pop, (char *) &node->header, cache_line_size, false, true);

            }
        }
        Node *head = D_RW(headoid);
        TOID(Node) nheadoid = headoid;
        nheadoid.oid.off = head->pairs[0].value;
        if (node == D_RW(nheadoid) && getNum(node->header) == 1 && !isBottom(node->header))
        {
            epoche->markNodeForDeletion((void *)head, threadEpocheInfo);
            head->header = (head->header | DEL_BITS);
            Node::clflush(pop, (char *)&head->header, sizeof(uint64_t), false, true);
            headoid = nheadoid;
            rootoid.oid.off = D_RW(headoid)->pairs[0].value;
            Node::clflush(pop, (char *)this, sizeof(SSBTree), false, true);
        }
    }




    //return upper-bound's location
    void SSBTree::linear_search(int &k, Pair *offset_pair, const int &n, const uint64_t &findkey)
    {
        if (k <= n)
            prefetch_((const char *)&offset_pair[k]);
        int m = n - 4;
        for (; k <= m ; k++)
        {
            prefetch_((const char *)&offset_pair[k + 4]);

            if (offset_pair[k].key > findkey)
                return;
            k++;
            if (offset_pair[k].key > findkey)
                return;
            k++;
            if (offset_pair[k].key > findkey)
                return;
            k++;
            if (offset_pair[k].key > findkey)
                return;
        }
        for (; k <= n ; k++)
            if (offset_pair[k].key > findkey)
                break;

        return;
    }

    void SSBTree::split(Node *node)
    {
        uint64_t header = node -> header;
        uint64_t num = getNum(header);
        if (num < maxPairsLength) return;
        int lazyflag = (header >> shiflazybox) & 3;

        uint64_t end = num - lazydiff[lazyflag] - 1 ;
        Pair *offset_pair = &node->pairs[0];
        if (versionTurn(header))
        {
            offset_pair = &node->pairs[maxPairsLength];
        }

        TOID(Node) newoid;
        pmemobj_xalloc(this->pop, &newoid.oid, sizeof(Node), 0, POBJ_CLASS_ID(128), NULL, NULL);
        Node *newnode = D_RW(newoid);

        pmemobj_mutex_zero(pop, &newnode->mutex);
        newnode ->right[rightTurn(header)] = node ->right[rightTurn(header)];
        newnode ->maxKey[rightTurn(header)] = node ->maxKey[rightTurn(header)];

        uint64_t newhead1, newhead2;

        uint64_t mid = (end + 1) >> 1;

        memcpy(newnode->pairs, offset_pair + mid , (end - mid + 1)*sizeof(Pair));
        newnode->midkey[0] = newnode->pairs[midindex].key;
        newhead1 = (header - ((end - mid + 1) << shifnumber)) ;
        Node::addRight(newhead1);
        newhead2 = (header & (~(LOCK_BITS | NUM_BITS | VERSION_BITS))) + ((end - mid + 1) << shifnumber); //unock&num

        if (lazyflag)
        {
            newnode->LazyBox = node ->LazyBox;
            if (newnode->LazyBox.key >= newnode->pairs[0].key)
            {
                uint64_t v = reinterpret_cast<uint64_t>(newnode->LazyBox.value);
                uint32_t w1 = ((v ^ signextend(v)) >> highPosition) - mid;
                newnode->LazyBox.value = signextend(v) ^ ((uint64_t)w1 << highPosition);

                newhead1 = newhead1 ^ (newhead1 & BOX_BITS);
                if (lazyflag == 0x1)
                {
                    newhead1 -= addNum_BITS;
                    newhead2 += addNum_BITS;
                }
                else
                {
                    newhead1 += addNum_BITS;
                    newhead2 -= addNum_BITS;
                }
            }
            else  newhead2 = newhead2 ^ (newhead2 & BOX_BITS);
        }

        newnode->header = (newhead2);
        Node::clflush(pop, (char *)newnode, NodeSize - maxPairsLength * sizeof(Pair) - sizeof(Pair) - cache_line_size, false, false);

        node->right[rightTurn(newhead1)] = newoid.oid.off;
        Node::clflush(pop, (char *)&node->right, sizeof(Pair), false, false);

        node->maxKey[rightTurn(newhead1)] = newnode->pairs[0].key;
        node->header = (newhead1);
        Node::clflush(pop, (char *)node, cache_line_size, true, true);
    }
    void SSBTree::merge(Node *node, ThreadInfo &threadEpocheInfo)
    {

        pmemobj_mutex_lock(pop, &node->mutex);

        uint64_t header = node -> header;
        if (isObsolete(header))
        {
            pmemobj_mutex_unlock(pop, &node->mutex);
            return;
        }

        TOID(Node) sibloid = headoid;
        sibloid.oid.off = node->right[rightTurn(header)];
        Node *sibling = D_RW(sibloid);
        pmemobj_mutex_lock(pop, &sibling->mutex);

        uint64_t sibling_header = sibling->header;

        if (isObsolete(sibling_header))
        {
            pmemobj_mutex_unlock(pop, &node->mutex);
            pmemobj_mutex_unlock(pop, &sibling->mutex);
            return;
        }

        if (getNum(header) + getNum(sibling_header) >= Lnum)
        {
            pmemobj_mutex_unlock(pop, &node->mutex);
            pmemobj_mutex_unlock(pop, &sibling->mutex);
            return;
        }

        int lazyflag1 = (header >> shiflazybox) & 3;
        int lazyflag2 = (sibling_header >> shiflazybox) & 3;
        int end1 = getNum(header) - 1 - lazydiff[lazyflag1];
        int oldend = end1;
        int end2 = getNum(sibling_header) - 1 - lazydiff[lazyflag2];
        Pair *offset_pair = &node->pairs[0];
        if (versionTurn(header))
        {
            offset_pair = &node->pairs[maxPairsLength];
        }

        Pair *sibling_pair  = &sibling->pairs[0];
        if (versionTurn(sibling_header))
        {
            sibling_pair = &sibling_pair[maxPairsLength];
        }
        int w = -1;
        uint64_t v = reinterpret_cast<uint64_t>(sibling->LazyBox.value);

        if (lazyflag2)
            w = (v ^ signextend(v)) >> highPosition;

        for (int i = 0 ; i < w; i++)
        {
            offset_pair[++end1] = sibling_pair[i];
        }

        if (lazyflag2 == 1) //insert
        {
            offset_pair[++end1].key = sibling->LazyBox.key;
            offset_pair[end1].value = signextend(v);
        }
        else w++;

        for (int i = w ; i <= end2; i++)
        {
            offset_pair[++end1] = sibling_pair[i];
        }

        Node::clflush(pop, (char *)offset_pair + oldend + 1, (end1 - oldend)*sizeof(Pair), false, false);

        uint64_t newheader = header  + addNum_BITS * (end1 - oldend);
        Node::addRight(newheader);

        node->right[rightTurn(newheader)] = sibling->right[rightTurn(sibling_header)];
        Node::clflush(pop, (char *)&node->right, cache_line_size, false, false);

        node->midkey[versionTurn(newheader)] = offset_pair[midindex].key;
        node->maxKey[rightTurn(newheader)] = sibling->maxKey[rightTurn(sibling_header)];
        node->header = newheader;
        Node::clflush(pop, (char *)&node->header, cache_line_size, true, true);

        epoche->markNodeForDeletion((void *)sibling, threadEpocheInfo);
        sibling->header = (sibling_header | DEL_BITS);
        //Node::clflush(pop,(char *)&sibling->header,sizeof(uint64_t),false,true);
        pmemobj_mutex_unlock(pop, &node->mutex);
        pmemobj_mutex_unlock(pop, &sibling->mutex);
    }

    void SSBTree::leafscan(TOID(Node) nodeoid, const uint64_t minscan, const uint64_t maxscan, int length, uint64_t *results, int &offset)
    {
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            int old_offset = offset;
            uint64_t header = node->header;
            int num = getNum(header) ;
            int lazyflag = (header >> shiflazybox) & 0x3;

            Pair lazybox = node->LazyBox;
            num -= lazydiff[lazyflag];

            Pair *offset_pair = &node->pairs[0];
            if (versionTurn(header))
            {
                offset_pair = &node->pairs[maxPairsLength];
            }

            int w = -1;
            uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);

            if (lazyflag)
                w = (v ^ signextend(v)) >> highPosition;

            for (int i = 0 ; i < w; i++)
                if (offset_pair[i].key >= minscan)
                {
                    if (offset_pair[i].key > maxscan) break;

                    results[offset++] = reinterpret_cast<uint64_t>(offset_pair[i].value);

                    if (offset == length) return;
                }
            if (lazyflag == 1 && lazybox.key >= minscan && lazybox.key <= maxscan)
            {
                results[offset++] = signextend(v);
                if (offset == length) return;
            }
            if (lazyflag != 1) w++;
            for (int i = w ; i < num; i++)
            {
                if (offset_pair[i].key >= minscan)
                {
                    if (offset_pair[i].key > maxscan) break;
                    results[offset++] = reinterpret_cast<uint64_t>(offset_pair[i].value);
                    if (offset == length) return;
                }
            }
            if (!Node::ReadcheckVesion(header, node->header))
                offset = old_offset;
            else nodeoid.oid.off = node->right[rightTurn(header)];
        }
    }


    /**************************************basic operators*******************************************************************/

    uint64_t SSBTree::lookup(const uint64_t findkey, ThreadInfo &threadEpocheInfo)
    {

        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid = rootoid;
        TOID(Node) nextoid = headoid;
        // Reading Process
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t header = node -> header;
            while ( node->maxKey[rightTurn(header)] <= findkey)
            {
                nodeoid.oid.off = node->right[rightTurn(header)];
                if (!Node::RightCheck(node->header, header))
                    goto restart;
                node = D_RW(nodeoid);
                header = node->header;
            }


            //handle lazybox


            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];

            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
            }

            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;
            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair upPair;


            //line search

            int k = 0;
            if (oldend >= midindex && midkey <= findkey)
            {
                k = midindex + 1;
                linear_search(k, offset_pair, oldend, findkey);
            }
            else linear_search(k, offset_pair, std::min(midindex, oldend), findkey);
            if (k == 0)
            {
                upPair.key = 0;
                nextoid.oid.off = upPair.value = 0;
            }
            else
            {
                upPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = upPair.value = offset_pair[k - 1].value;
            }

            k--;


            //compare with lazybox
            uint64_t succKey = node -> maxKey[rightTurn(header)];

            if (k < oldend )
            {
                uint64_t nextKey =  offset_pair[k + 1].key;
                if (lazyflag == 0x2  && lazybox.key == nextKey)
                {
                    if (k + 1 < oldend)
                        succKey = offset_pair[k + 2].key;
                }
                else succKey = nextKey;

            }


            if (lazyflag == 0x2  && lazybox.key == upPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    upPair.key = 0;
                    nextoid.oid.off = upPair.value = 0;
                }
            }

            if (lazyflag == 1)
            {
                if (lazybox.key <= findkey && lazybox.key >= upPair.key)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    upPair.value = nextoid.oid.off = signextend(v);
                    upPair.key = lazybox.key;
                }

                if (lazybox.key > findkey && lazybox.key <= succKey)
                    succKey = lazybox.key;

            }



            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;

            if (isBottom(header))
            {
                if (upPair.key != findkey) return 0;
                return reinterpret_cast<uint64_t> (upPair.value);
            }


            //3.Horizontal traversal

            bool op = true;//up;
            bool needdown = false;

#ifdef REBALANCE
            TOID(Node) iteratoroid = nextoid;
            Node *iterator = D_RW(nextoid);
            TOID(Node) temp = nextoid;
            uint32_t sum = 0;
            uint64_t node_upper = node->maxKey[rightTurn(header)];
            //down
            uint64_t htd = iterator->header;
            temp.oid.off = iterator->right[rightTurn(htd)];
            sum += getNum(htd) + getNum(D_RW(temp)->header);
            if ( sum < Lnum && succKey != (uint64_t) - 1 && !(iterator->maxKey[rightTurn(htd)] == succKey && succKey == node_upper) )
            {
                if (succKey != node_upper && iterator->maxKey[rightTurn(htd)] == succKey)
                {
                    needdown = true;
                    while (k < oldend && offset_pair[k + 1].key <= succKey)
                        k++;
                }
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                //down
                op = false;
                upPair.key = succKey;
                goto WriteProcess;
            }

            sum = 0;
            htd = iterator->header;

            while (iterator->maxKey[rightTurn(htd)] < findkey)
            {
                iteratoroid.oid.off = iterator->right[rightTurn(htd)];
                sum += getNum(iterator->header);
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iteratoroid.oid.off;
                //if (iterator->maxKey[rightTurn(htd)]<findkey)
                nextoid = iteratoroid;
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                iterator = D_RW(iteratoroid);
                htd = iterator->header;

            }
            sum += getNum(htd);
            // No need to up
            if (sum <= Rnum )
            {
                nodeoid = nextoid;
                continue;
            }
#else

            Node *iterator = D_RW(nextoid);
            uint64_t htd = iterator->header;
            if (iterator->maxKey[rightTurn(htd)] < succKey)
            {
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iterator->right[rightTurn(htd)];
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;
            }
            else
            {
                nodeoid = nextoid;
                continue;
            }

#endif

            //up


            //4.Writing Process
WriteProcess:

            //pmemobj_mutex_lock(pop,&node->mutex);

            if (pmemobj_mutex_trylock(pop, &node->mutex))
            {
                nodeoid = nextoid;
                continue;
            }
            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }
            if (op)
            {
                upKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend);
                pmemobj_mutex_unlock(pop, &node->mutex);
            }
            else
            {
                if (needdown)
                    downKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
                pmemobj_mutex_unlock(pop, &node->mutex);
                merge(D_RW(nextoid), threadEpocheInfo);
            }
            nodeoid = nextoid;
        }
        return 0;
    }

    void SSBTree::put(const uint64_t insertKey, const uint64_t insertValue, ThreadInfo &threadEpocheInfo)
    {
        assert(insertKey != 0);
        assert(insertKey != (uint64_t) - 1);
        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid = headoid;
        TOID(Node) nextoid = headoid;
        // Reading Process
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t header = node -> header;
            while ( node->maxKey[rightTurn(header)] <= insertKey)
            {
                nodeoid.oid.off = node->right[rightTurn(header)];
                if (!Node::RightCheck(node->header, header))
                    goto restart;
                node = D_RW(nodeoid);
                header = node->header;
            }



            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];
            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
            }

            uint64_t succKey = node -> maxKey[rightTurn(header)];
            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;
            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair upPair;

            int k = 0;
            if (oldend >= midindex && midkey <= insertKey)
                k = midindex + 1;
            linear_search(k, offset_pair, oldend, insertKey);

            if (k == 0)
            {
                upPair.key = 0;
                nextoid.oid.off = upPair.value = 0;
            }
            else
            {
                upPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = upPair.value = offset_pair[k - 1].value;
            }

            k--;
            //compare with lazybox
            if (k < oldend )
            {
                uint64_t nextKey =  offset_pair[k + 1].key;
                if (lazyflag == 0x2  && lazybox.key == nextKey)
                {
                    if (k + 1 < oldend)
                        succKey = offset_pair[k + 2].key;
                }
                else succKey = nextKey;

            }

            if (lazyflag == 0x2  && lazybox.key == upPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    upPair.key = 0;
                    nextoid.oid.off = upPair.value = insertValue;
                }
            }

            if (lazyflag == 1)
            {
                if (lazybox.key <= insertKey && lazybox.key >= upPair.key)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    nextoid.oid.off = signextend(v);
                }
                if (lazybox.key > insertKey && lazybox.key <= succKey)
                    succKey = lazybox.key;
            }


            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;


            bool op = true;//up;
            bool needdown = false;
            Node *iterator = D_RW(nextoid);
            TOID(Node) temp = nextoid;
            TOID(Node) iteratoroid = nextoid;
            uint32_t sum = 0;
            uint64_t node_upper = node->maxKey[rightTurn(header)];
            uint64_t htd = 0;
            if (isBottom(header) )
            {
                upPair.key = insertKey;
                upPair.value = insertValue;
                goto WriteProcess;
            }


            //3.Horizontal traversal

#ifdef REBALANCE
            //down
            htd = iterator->header;
            temp.oid.off = iterator->right[rightTurn(htd)];
            sum += getNum(htd) + getNum(D_RW(temp)->header);
            if ( sum < Lnum && succKey != (uint64_t) - 1 && !(iterator->maxKey[rightTurn(htd)] == succKey && succKey == node_upper) )
            {
                if (succKey != node_upper && iterator->maxKey[rightTurn(htd)] == succKey)
                {
                    needdown = true;
                    while (k < oldend && offset_pair[k + 1].key <= succKey)
                        k++;
                }
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                //down
                op = false;
                upPair.key = succKey;
                goto WriteProcess;
            }

            //up
            sum = 0;
            htd = iterator->header;

            while (iterator->maxKey[rightTurn(htd)] < insertKey)
            {
                iteratoroid.oid.off = iterator->right[rightTurn(htd)];
                sum += getNum(iterator->header);
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iteratoroid.oid.off;
                //if (iterator->maxKey[rightTurn(htd)]<insertKey)
                nextoid = iteratoroid;
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                iterator = D_RW(iteratoroid);
                htd = iterator->header;
            }
            sum += getNum(htd);
            // No need to up
            if (sum <= Rnum )
            {
                nodeoid = nextoid;
                continue;
            }

#else
            htd = iterator->header;
            if (iterator->maxKey[rightTurn(htd)] < succKey)
            {
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iterator->right[rightTurn(htd)];
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;
            }
            else
            {
                nodeoid = nextoid;
                continue;
            }
#endif

            //4.Writing Process
WriteProcess:

            //pmemobj_mutex_lock(pop,&node->mutex);

            if (!isBottom(header))
            {
                if (pmemobj_mutex_trylock(pop, &node->mutex))
                {
                    nodeoid = nextoid;
                    continue;
                }
            }
            else pmemobj_mutex_lock(pop, &node->mutex);

            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }

            if (op)
            {
                upKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend);
                pmemobj_mutex_unlock(pop, &node->mutex);
            }
            else
            {
                if (needdown)
                    downKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
                pmemobj_mutex_unlock(pop, &node->mutex);
                merge(D_RW(nextoid), threadEpocheInfo);
            }
            if (isBottom(header)) return;
            nodeoid = nextoid;
        }
    }

    void SSBTree::update(const uint64_t updatekey, const uint64_t updatevalue, ThreadInfo &threadEpocheInfo)
    {

        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid = rootoid;
        TOID(Node) nextoid = headoid;
        // Reading Process
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t header = node -> header;
            while ( node->maxKey[rightTurn(header)] <= updatekey)
            {
                nodeoid.oid.off = node->right[rightTurn(header)];
                if (!Node::RightCheck(node->header, header))
                    goto restart;
                node = D_RW(nodeoid);
                header = node->header;
            }

            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];
            Oidoff *needupdate = &node->pairs[0].value;
            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
                needupdate = &node->pairs[maxPairsLength].value;
            }

            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;
            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair upPair;

            //line search
            int k = 0;
            if (oldend >= midindex && midkey <= updatekey)
                k = midindex + 1;
            linear_search(k, offset_pair, oldend, updatekey);
            if (k)
            {
                upPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = upPair.value = offset_pair[k - 1].value;
                needupdate =  &offset_pair[k - 1].value;
            }
            k--;
            //compare with lazybox
            uint64_t succKey = node -> maxKey[rightTurn(header)];

            if (k < oldend )
            {
                uint64_t nextKey =  offset_pair[k + 1].key;
                if (lazyflag == 0x2  && lazybox.key == nextKey)
                {
                    if (k + 1 < oldend)
                        succKey = offset_pair[k + 2].key;
                }
                else succKey = nextKey;

            }

            if (lazyflag == 0x2  && lazybox.key == upPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    upPair.key = 0;
                    nextoid.oid.off = upPair.value = 0;
                    needupdate =  0;
                }
            }

            if (lazyflag == 1)
            {
                if (lazybox.key <= updatekey && lazybox.key >= upPair.key)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    upPair.value = nextoid.oid.off = signextend(v);
                    upPair.key = lazybox.key;
                    needupdate = &node->LazyBox.value;
                }
                if (lazybox.key > updatekey && lazybox.key <= succKey)
                    succKey = lazybox.key;
            }


            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;

            if (isBottom(header) )
            {
                if (upPair.key != updatekey) return;
                pmemobj_mutex_lock(pop, &node->mutex);
                //checkversion
                if (!Node::ReadcheckVesion(header, node->header) || isObsolete(header))
                {
                    pmemobj_mutex_unlock(pop, &node->mutex);
                    goto restart;
                }

                if (upPair.key == lazybox.key && lazyflag)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    uint32_t w = (v ^ signextend(v)) >> highPosition;
                    *needupdate = updatevalue ^ ((uint64_t)w << highPosition);
                }
                else
                {
                    *needupdate = updatevalue;
                }
                pmemobj_mutex_unlock(pop, &node->mutex);
                return ;
            }

            bool op = true;//up;
            bool needdown = false;
            //3.Horizontal traversal

#ifdef REBALANCE
            //down
            TOID(Node) temp = nextoid;
            TOID(Node) iteratoroid = nextoid;
            Node *iterator = D_RW(nextoid);
            uint32_t sum = 0;
            uint64_t node_upper = node->maxKey[rightTurn(header)];
            uint64_t htd = iterator->header;
            temp.oid.off = iterator->right[rightTurn(htd)];
            sum += getNum(htd) + getNum(D_RW(temp)->header);
            if ( sum < Lnum && succKey != (uint64_t) - 1 && !(iterator->maxKey[rightTurn(htd)] == succKey && succKey == node_upper) )
            {
                if (succKey != node_upper && iterator->maxKey[rightTurn(htd)] == succKey)
                {
                    needdown = true;
                    while (k < oldend && offset_pair[k + 1].key <= succKey)
                        k++;
                }
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                //down
                op = false;
                upPair.key = succKey;
                goto WriteProcess;
            }

            //up
            sum = 0;
            htd = iterator->header;

            while (iterator->maxKey[rightTurn(htd)] < updatekey)
            {
                iteratoroid.oid.off = iterator->right[rightTurn(htd)];
                sum += getNum(iterator->header);
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iteratoroid.oid.off;
                //if (iterator->maxKey[rightTurn(htd)]<updatekey)
                nextoid = iteratoroid;
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;
                iterator = D_RW(iteratoroid);
                htd = iterator->header;
            }
            sum += getNum(htd);
            // No need to up
            if (sum <= Rnum )
            {
                nodeoid = nextoid;
                continue;
            }

#else

            Node *iterator = D_RW(nextoid);
            uint64_t htd = iterator->header;
            if (iterator->maxKey[rightTurn(htd)] < succKey)
            {
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iterator->right[rightTurn(htd)];
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

            }
            else
            {
                nodeoid = nextoid;
                continue;
            }
#endif

            //4.Writing Process
WriteProcess:

            //pmemobj_mutex_lock(pop,&node->mutex);
            if (pmemobj_mutex_trylock(pop, &node->mutex))
            {
                nodeoid = nextoid;
                continue;
            }
            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }
            header = node->header;
            if (op)
            {
                upKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend);
                pmemobj_mutex_unlock(pop, &node->mutex);
            }
            else
            {
                if (needdown)
                    downKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
                pmemobj_mutex_unlock(pop, &node->mutex);
                merge(D_RW(nextoid), threadEpocheInfo);
            }
            nodeoid = nextoid;
        }
    }

    void SSBTree::normalRemove(const uint64_t removekey, ThreadInfo &threadEpocheInfo)
    {
        assert(removekey != 0);
        assert(removekey != (uint64_t) - 1);
        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid =  headoid;
        TOID(Node) nextoid = headoid;
        // Reading Process
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t header = node -> header;
            while ( node->maxKey[rightTurn(header)] <= removekey)
            {
                nodeoid.oid.off = node->right[rightTurn(header)];
                if (!Node::RightCheck(node->header, header))
                    goto restart;
                node = D_RW(nodeoid);
                header = node->header;
            }
            //if (debug==35) return;
            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];

            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
            }

            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;
            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair downPair;

            //line search
            int k = 0;
            if (oldend >= midindex && midkey <= removekey)
                k = midindex + 1;
            linear_search(k, offset_pair, oldend, removekey);
            if (k == 0)
            {
                downPair.key = 0;
                nextoid.oid.off = downPair.value = 0;
            }
            else
            {
                downPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = downPair.value = offset_pair[k - 1].value;
            }
            k--;
            //compare with lazybox

            if (lazyflag == 0x2  && lazybox.key == downPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    downPair.key = 0;
                    nextoid.oid.off = downPair.value = 0;
                }
            }

            if (lazyflag == 1)
            {

                if (lazybox.key <= removekey && lazybox.key > downPair.key)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    nextoid.oid.off  = signextend(v);
                }
            }

            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;

            if (!isBottom(header) )
            {
                nodeoid = nextoid;
                continue;
            }
            //4.Writing Process
            if (downPair.key != removekey && (!lazyflag || removekey != lazybox.key)) return;
            downPair.key = removekey;

            pmemobj_mutex_lock(pop, &node->mutex);

            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }
            header = node->header;
            downKey(node, header, lazyflag, lazybox, downPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
            pmemobj_mutex_unlock(pop, &node->mutex);
            return;
        }
    }

    void SSBTree::balanceRemove(const uint64_t removekey, ThreadInfo &threadEpocheInfo)
    {
        assert(removekey != 0);
        assert(removekey != (uint64_t) - 1);
        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid = headoid;
        TOID(Node) nextoid = headoid;

        // Reading Process
        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t htd = node->header;
            while (node->maxKey[rightTurn(htd)] <= removekey)
            {
                nodeoid.oid.off = node->right[rightTurn(htd)];
                if (!Node::RightCheck(node->header, htd))
                    goto restart;
                node = D_RW(nodeoid);
                htd = node->header;
            }

            uint64_t header = node -> header;
            uint64_t succKey = node -> maxKey[rightTurn(header)];

            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];
            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
            }


            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;
            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair downPair;

            //line search

            int k = 0;
            if (oldend >= midindex && midkey <= removekey)
                k = midindex + 1;
            linear_search(k, offset_pair, oldend, removekey);

            if (k == 0)
            {
                downPair.key = 0;
                nextoid.oid.off = downPair.value = 0;
            }
            else
            {
                downPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = downPair.value = offset_pair[k - 1].value;
            }

            k--;

            //compare with lazybox
            uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);

            if (k < oldend )
            {
                uint64_t nextKey =  offset_pair[k + 1].key;
                if (lazyflag == 0x2  && lazybox.key == nextKey)
                {
                    if (k + 1 < oldend)
                        succKey = offset_pair[k + 2].key;
                }
                else succKey = nextKey;

            }
            if (lazyflag == 0x2  && lazybox.key == downPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    downPair.key = 0;
                    nextoid.oid.off = downPair.value = 0;
                }
            }

            if (lazyflag == 1)
            {

                if (lazybox.key <= removekey && lazybox.key > downPair.key)
                    nextoid.oid.off = signextend(v);

                if (lazybox.key > removekey && lazybox.key <= succKey)
                    succKey = lazybox.key;

            }
            TOID(Node) iteratoroid = nextoid;
            Node *iterator = D_RW(iteratoroid);
            TOID(Node) temp = nextoid;
            uint32_t sum = 0;
            uint64_t node_upper = node->maxKey[rightTurn(header)];
            bool op = false;//down;
            bool needdown = false;
            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;

            if (isBottom(header) )
            {
                if (downPair.key != removekey && (!lazyflag || removekey != lazybox.key)) return;
                downPair.key = removekey;
                needdown = true;
                goto WriteProcess;
            }


            //down
            htd = iterator->header;
            temp.oid.off = iterator->right[rightTurn(htd)];
            sum += getNum(htd) + getNum(D_RW(temp)->header);
            if ( sum < Lnum && succKey != (uint64_t) - 1 && !(iterator->maxKey[rightTurn(htd)] == succKey && succKey == node_upper))
            {
                if (succKey != node_upper && iterator->maxKey[rightTurn(htd)] == succKey)
                {
                    needdown = true;
                    while (k < oldend && offset_pair[k + 1].key <= succKey)
                        k++;
                }
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;
                //down
                downPair.key = succKey;
                goto WriteProcess;
            }

            //up
            sum = 0;
            htd = iterator->header;

            while (iterator->maxKey[rightTurn(htd)] < removekey)
            {
                iteratoroid.oid.off = iterator->right[rightTurn(htd)];
                sum += getNum(iterator->header);
                downPair.key =  iterator->maxKey[rightTurn(htd)];
                downPair.value =  iteratoroid.oid.off;
                //if (iterator->maxKey[rightTurn(htd)]<removekey)
                nextoid = iteratoroid;
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;
                iterator = D_RW(iteratoroid);
                htd = iterator->header;
            }
            sum += getNum(htd);
            // No need to up

            if (sum <= Rnum )
            {
                nodeoid = nextoid;
                continue;
            }

            op = true;


            //4.Writing Process
WriteProcess:

            //pmemobj_mutex_lock(pop,&node->mutex);
            if (!isBottom(header))
            {
                if (pmemobj_mutex_trylock(pop, &node->mutex))
                {
                    nodeoid = nextoid;
                    continue;
                }
            }
            else pmemobj_mutex_lock(pop, &node->mutex);
            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }
            header = node->header;
            if (op)
            {
                upKey(node, header, lazyflag, lazybox, downPair, move_pair, offset_pair, k, oldend);
                pmemobj_mutex_unlock(pop, &node->mutex);
            }
            else
            {
                if (needdown)
                    downKey(node, header, lazyflag, lazybox, downPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
                pmemobj_mutex_unlock(pop, &node->mutex);
                if (!isBottom(header))
                    merge(D_RW(nextoid), threadEpocheInfo);
            }
            if (isBottom(header)) return;
            nodeoid = nextoid;
        }
    }

    void SSBTree::remove(const uint64_t removekey, ThreadInfo &threadEpocheInfo)
    {
#ifdef REBALANCE
        balanceRemove(removekey, threadEpocheInfo);
#else
        normalRemove(removekey, threadEpocheInfo);
#endif
    }

    void SSBTree::scan(const uint64_t minscan, const uint64_t maxscan, int length, uint64_t *results, int &offset, ThreadInfo &threadEpocheInfo)
    {
        EpocheGuard epocheGuard(threadEpocheInfo);
restart:
        TOID(Node) nodeoid = rootoid;
        TOID(Node) nextoid = headoid;

        while(nodeoid.oid.off != tailoid.oid.off)
        {
            Node *node = D_RW(nodeoid);
            uint64_t header = node -> header;

            while ( node->maxKey[rightTurn(header)] <= minscan)
            {
                nodeoid.oid.off = node->right[rightTurn(header)];
                if (!Node::RightCheck(node->header, header))
                    goto restart;
                node = D_RW(nodeoid);
                header = node->header;
            }

            if (isBottom(header))
            {
                leafscan(nodeoid, minscan, maxscan, length, results, offset);
                return;
            }

            Pair *offset_pair = &node->pairs[0];
            Pair *move_pair = &node->pairs[maxPairsLength];
            uint64_t midkey = node -> midkey[0];
            if (versionTurn(header))
            {
                Pair *temp = offset_pair;
                offset_pair = move_pair;
                move_pair = temp;
                midkey = node -> midkey[1];
            }

            //line search

            int lazyflag = (header >> shiflazybox) & 3;
            Pair lazybox = node->LazyBox;

            int oldend = getNum(header) - 1 - lazydiff[lazyflag];
            Pair upPair;

            int k = 0;
            if (oldend >= midindex && midkey <= minscan)
                k = midindex + 1;
            linear_search(k, offset_pair, oldend, minscan);

            if (k == 0)
            {
                upPair.key = 0;
                nextoid.oid.off = upPair.value = 0;
            }
            else
            {
                upPair.key = offset_pair[k - 1].key;
                nextoid.oid.off = upPair.value = offset_pair[k - 1].value;
            }

            k--;
            //compare with lazybox
            uint64_t succKey = node -> maxKey[rightTurn(header)];

            if (k < oldend )
            {
                uint64_t nextKey =  offset_pair[k + 1].key;
                if (lazyflag == 0x2  && lazybox.key == nextKey)
                {
                    if (k + 1 < oldend)
                        succKey = offset_pair[k + 2].key;
                }
                else succKey = nextKey;

            }

            if (lazyflag == 0x2  && lazybox.key == upPair.key)
            {
                if (k > 0)  nextoid.oid.off = offset_pair[k - 1].value;
                else
                {
                    upPair.key = 0;
                    nextoid.oid.off = upPair.value = 0;
                }
            }

            if (lazyflag == 1)
            {
                if (lazybox.key <= minscan && lazybox.key >= upPair.key)
                {
                    uint64_t v = reinterpret_cast<uint64_t>(lazybox.value);
                    upPair.value = nextoid.oid.off = signextend(v);
                    upPair.key = lazybox.key;
                }
                if (lazybox.key > minscan && lazybox.key <= succKey)
                    succKey = lazybox.key;
            }


            bool op = true;//up;
            bool needdown = false;

            if (!Node::ReadcheckVesion(header, node->header))
                goto restart;

            //3.Horizontal traversal

#ifdef REBALANCE
            //down
            TOID(Node) temp = nextoid;
            TOID(Node) iteratoroid = nextoid;
            Node *iterator = D_RW(nextoid);
            uint32_t sum = 0;
            uint64_t node_upper = node->maxKey[rightTurn(header)];
            uint64_t htd = iterator->header;
            temp.oid.off = iterator->right[rightTurn(htd)];
            sum += getNum(htd) + getNum(D_RW(temp)->header);
            if ( sum < Lnum && succKey != (uint64_t) - 1 && !(iterator->maxKey[rightTurn(htd)] == succKey && succKey == node_upper) )
            {
                if (succKey != node_upper && iterator->maxKey[rightTurn(htd)] == succKey)
                {
                    needdown = true;
                    while (k < oldend && offset_pair[k + 1].key <= succKey)
                        k++;
                }
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                //down
                op = false;
                upPair.key = succKey;
                goto WriteProcess;
            }

            //up
            sum = 0;
            htd = iterator->header;

            while (iterator->maxKey[rightTurn(htd)] < minscan)
            {
                iteratoroid.oid.off = iterator->right[rightTurn(htd)];
                sum += getNum(iterator->header);
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iteratoroid.oid.off;
                //if (iterator->maxKey[rightTurn(htd)]<minscan)
                nextoid = iteratoroid;
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

                iterator = D_RW(iteratoroid);
                htd = iterator->header;


            }
            sum += getNum(htd);
            // No need to up
            if (sum <= Rnum )
            {
                nodeoid = nextoid;
                continue;
            }

#else
            Node *iterator = D_RW(nextoid);
            uint64_t htd = iterator->header;

            if (iterator->maxKey[rightTurn(htd)] < succKey)
            {
                upPair.key =  iterator->maxKey[rightTurn(htd)];
                upPair.value =  iterator->right[rightTurn(htd)];
                if (!Node::RightCheck(iterator->header, htd))
                    goto restart;

            }
            else
            {
                nodeoid = nextoid;
                continue;
            }
#endif
            //4.Writing Process
WriteProcess:

            //pmemobj_mutex_lock(pop,&node->mutex);

            if (pmemobj_mutex_trylock(pop, &node->mutex))
            {
                nodeoid = nextoid;
                continue;
            }
            //checkversion
            if (!Node::WritecheckVesion(header, node->header) || isObsolete(header))
            {
                pmemobj_mutex_unlock(pop, &node->mutex);
                goto restart;
            }
            header = node->header;
            if (op)
            {
                upKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend);
                pmemobj_mutex_unlock(pop, &node->mutex);
            }
            else
            {
                if (needdown)
                    downKey(node, header, lazyflag, lazybox, upPair, move_pair, offset_pair, k, oldend, threadEpocheInfo);
                pmemobj_mutex_unlock(pop, &node->mutex);
                merge(D_RW(nextoid), threadEpocheInfo);
            }
            nodeoid = nextoid;
        }
    }

}




