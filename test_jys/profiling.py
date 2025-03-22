# cProfile
import pstats

p = pstats.Stats('profile.stats')
p.sort_stats('cumulative')
p.print_stats(30)
p.print_callers(2)
p.print_callees(1)

# bytecode
import dis
import test_jys.bytecode

dis.dis(test_jys.bytecode.forward)