import pandas as pd
import numpy as np
def insert_sort(nums):
    if nums is None:
        return nums
    l=len(nums)
    for i in range(1,l):
        temp=nums[i]
        j=i
        while j>0 and nums[j-1]>=temp:
            nums[j]=nums[j-1]
            j-=1
        nums[j]=temp
    return nums

def quick_sort(nums,l,h):
    if nums is None or l>=h:
        return None
    mid=nums[h]
    less=l-1
    more=l-1
    i=l
    while i<=h:
        if nums[i]<=mid:
            less+=1
            temp=nums[i]
            nums[i]=nums[less]
            nums[less]=temp
        more+=1
        i+=1
    quick_sort(nums,l,less-1)
    quick_sort(nums,less+1,h)

def merge_sort(nums,p,r):
    if p>=r:
        return None
    q=(p+r)/2
    merge_sort(nums,p,q)
    merge_sort(nums,q+1,r)
    merge(nums,p,q,r)

def merge(nums,p,q,r):
    arr_l=nums[p:q+1]
    arr_r=nums[q+1:r+1]
    arr_l.append(np.float('inf'))
    arr_r.append(np.float('inf'))
    i=p
    c1=0
    c2=0
    while i<=r:
        if arr_l[c1]>arr_r[c2]:
            nums[i]=arr_r[c2]
            c2+=1
        else:
            nums[i]=arr_l[c1]
            c1+=1
        i+=1





if __name__=='__main__':
    nums=np.random.randint(1,100,30)
    print nums
    nums=list(nums)
    merge_sort(nums,0,29)
    print nums

