from jug import Task, TaskGenerator
@TaskGenerator
def print_detailed_results(results):
    import numpy as np
    ds = [d for d,_,_ in results]
    ds = set(ds)
    methods = set(m for _,n,m in results)
    norms = set(n for _,n,_ in results)
    with open('results/by_data.txt', 'w') as output:
        for d in ds:
            print >>output, d
            res = []
            for m in methods:
                if m.startswith('random('):
                    continue
                for n in norms:
                    r = np.mean(results[d,n,m][0])
                    print >>output, '{1}-{0:24} {2:.2%}'.format(m,n,r)
                    res.append(r)
            print >>output
            print >>output, '{0:24} {1:.2%} {2:.2%} {3:.2%}'.format('mean/median/max', np.mean(res), np.median(res), np.max(res))
            print >>output
            print >>output
            print >>output
            
    with open('results/by_norm.txt', 'w') as output:
        for n in norms:
            print >>output, n
            res = []
            for d in ds:
                for m in methods:
                    if m.startswith('random('):
                        continue
                    r = np.mean(results[d,n,m][0])
                    print >>output, '{0}-{1:24} {2:.2%}'.format(d,m,r)
                    res.append(r)
            print >>output
            print >>output, '{0:24} {1:.2%} {2:.2%} {3:.2%}'.format('mean/median/max', np.mean(res), np.median(res), np.max(res))
            print >>output
            print >>output
            print >>output
            
            
    with open('results/by_method.txt', 'w') as output:
        for m in methods:
            if m.startswith('random('):
                continue
            print >>output, m
            res = []
            for d in ds:
                for n in norms:
                    r = np.mean(results[d,n,m][0])
                    print >>output, '{0}-{1:24} {2:.2%}'.format(d,n,r)
                    res.append(r)
            print >>output
            print >>output, '{0:24} {1:.2%} {2:.2%} {3:.2%}'.format('mean/median/max', np.mean(res), np.median(res), np.max(res))
            print >>output
            print >>output
            print >>output
            
    with open('results/by_method_norm.txt', 'w') as output:
        for m in methods:
            if m.startswith('random('):
                continue
            for n in norms:
                print >>output, '{0}-{1}'.format(m,n)
                res = []
                for d in ds:
                    r = np.mean(results[d,n,m][0])
                    print >>output, '{0:24} {1:.2%}'.format(d,r)
                    res.append(r)
                print >>output
                print >>output, '{0:24} {1:.2%} {2:.2%} {3:.2%}'.format('mean/median/max', np.mean(res), np.median(res), np.max(res))
                print >>output
                print >>output
                print >>output
            
