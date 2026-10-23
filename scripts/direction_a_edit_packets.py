"""Valid matching edits and a truth-using diagnostic oracle, not a predictor."""

def valid_matching(pairs):
    ends=[v for a,b in pairs for v in (a,b)]
    return all(0<=a<b for a,b in pairs) and len(ends)==len(set(ends))

def edit_packets(base,proposal,merge_stems=False):
    base,proposal=set(map(tuple,base)),set(map(tuple,proposal));assert valid_matching(base) and valid_matching(proposal)
    removed=base-proposal;added=proposal-base;difference=removed|added;adjacency={}
    for a,b in difference:adjacency.setdefault(a,set()).add(b);adjacency.setdefault(b,set()).add(a)
    components=[];seen=set()
    for start in sorted(adjacency):
        if start in seen:continue
        nodes=set();todo=[start]
        while todo:
            value=todo.pop()
            if value in nodes:continue
            nodes.add(value);todo.extend(adjacency[value]-nodes)
        seen|=nodes;components.append(nodes)
    packets=[dict(removed=sorted(p for p in removed if p[0] in nodes),added=sorted(p for p in added if p[0] in nodes)) for nodes in components]
    if not merge_stems:return packets
    parent=list(range(len(packets)));owner={p:i for i,row in enumerate(packets) for p in row['removed']+row['added']}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for structure in (base,proposal):
        for a,b in sorted(structure):
            if (a-1,b+1) in structure:continue
            ids=[];pair=(a,b)
            while pair in structure:
                if pair in owner:ids.append(owner[pair])
                pair=(pair[0]+1,pair[1]-1)
            if ids:
                for i in ids[1:]:parent[find(i)]=find(ids[0])
    merged={}
    for i,row in enumerate(packets):
        target=merged.setdefault(find(i),dict(removed=[],added=[]))
        for key in ('removed','added'):target[key].extend(row[key])
    return [dict(removed=sorted(row['removed']),added=sorted(row['added'])) for _,row in sorted(merged.items())]

def oracle(base,proposal,truth,merge_stems=True):
    """Exact F1 maximum over independent packets by integer fractional optimization.

    For a current fraction N/D, choose packet c iff 2*deltaTP_c*D -
    N*(deltaTP_c+deltaFP_c)>0. Packets have disjoint endpoints; any subset is
    a valid matching. Labels are used directly: this is only an oracle bound.
    """
    base,proposal,truth=map(lambda p:set(map(tuple,p)),(base,proposal,truth));packets=edit_packets(base,proposal,merge_stems)
    tp=len(base&truth);fp=len(base-truth);numerator=2*tp;denominator=len(base)+len(truth)
    if denominator==0:return dict(prediction=sorted(base),f1=0.,packets=len(packets),accepted=0,uses_truth=True)
    changes=[]
    for packet in packets:
        removed,added=map(set,(packet['removed'],packet['added']))
        changes.append((len(added&truth)-len(removed&truth),len(added-truth)-len(removed-truth)))
    for _ in range(len(packets)+3):
        chosen=[i for i,(dtp,dfp) in enumerate(changes) if 2*dtp*denominator-numerator*(dtp+dfp)>0]
        new_tp=tp+sum(changes[i][0] for i in chosen);new_fp=fp+sum(changes[i][1] for i in chosen)
        new_n=2*new_tp;new_d=new_tp+new_fp+len(truth)
        assert new_d>0 and new_n*denominator>=numerator*new_d
        if new_n*denominator==numerator*new_d:break
        numerator,denominator=new_n,new_d
    else:raise RuntimeError('Oracle did not converge')
    prediction=set(base)
    for i in chosen:prediction.difference_update(packets[i]['removed']);prediction.update(packets[i]['added'])
    assert valid_matching(prediction)
    value=2*len(prediction&truth)/(len(prediction)+len(truth)) if prediction or truth else 0.
    return dict(prediction=sorted(prediction),f1=value,packets=len(packets),accepted=len(chosen),uses_truth=True)
