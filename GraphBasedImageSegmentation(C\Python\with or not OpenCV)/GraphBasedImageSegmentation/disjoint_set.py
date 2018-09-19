# -*- coding:utf-8 -*-

class universe:
    def __init__(self,elements):
        self.elts = []
        self.num = elements
        for i in range(elements):
            # rank,size,p
            self.elts.append([0, 1, i])

    def get_size(self,x):
        return self.elts[x][1]

    # 找到并且最小化
    def find(self,x):
        x = int(x)
        y = x
        while y != self.elts[y][2]:
            y = self.elts[y][2]
            self.elts[x][2] = y
        return y

    # 加入
    def join(self,x,y):
        if self.elts[x][0] > self.elts[y][0]:
            self.elts[y][2] = x
            self.elts[x][1] += self.elts[y][1]
        else:
            self.elts[x][2] = y
            self.elts[y][1] += self.elts[x][1]
            if self.elts[x][0] == self.elts[y][0]:
                self.elts[y][1] += 1
        self.num += -1
