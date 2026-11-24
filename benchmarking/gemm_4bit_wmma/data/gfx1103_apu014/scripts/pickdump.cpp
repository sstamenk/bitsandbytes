// pickdump: the tile every rule table of a target picks, for "N K" pairs on stdin and M = 1..600 (verifies the python
// candidate files against the C++ header).
// the rules header is passed with -include (a stale copy sits next to this file)
#include <cstdio>
template <int R> void dump(const char* name, const WmmaRule (&rules)[R]) {
    std::rewind(stdin);
    int N, K;
    while (std::scanf("%d %d", &N, &K) == 2)
        for (int M = 1; M <= 600; M++) {
            WmmaTile t = wmma_pick(rules, M, N, K);
            std::printf("%s %d %d %d %d,%d,%d,%d,%d,%d\n", name, M, N, K, t.wm, t.wn, t.warps_m, t.warps_n, t.ksplit, t.kch);
        }
}
int main() {
    dump("g1103_f16", WmmaTileRules<1103>::f16);
    dump("g1103_f32", WmmaTileRules<1103>::f32);
    dump("g1100_f16", WmmaTileRules<1100>::f16);
    dump("g1100_f32", WmmaTileRules<1100>::f32);
}
