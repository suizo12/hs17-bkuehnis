def get_score(s, l):
    """
    get the porcentage of value s in relation of the values of
    :param s:
    :param l:
    :return:
    """

    l = list(l)
    l = sorted(l)

    if s in l:
        return round(l.index(s) * 100 / len(l) )
    return None

def get_percent_score(s, l):
    l = list(l)
    l = sorted(l)
    print(l)
    median = sum(l) / len(l)
    #median = 1000
    #print(int(s * 100 / l[len(l)-1]))
    r = int(s * 50 / median)
    #print(r, s)
    return r
