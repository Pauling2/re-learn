#coding=utf8


##目的：统计每个底物能被多少种激酶磷酸化，并对磷酸化统一位点的激酶进行分类

##############python2中，如果传入的参数为地址，地址中的\可以正常识别；python3中必须加上转义r###############
import re


def open_multi(filename1,filename2,specify):
#################统计出所有的不同位点#####################	
	with open(filename1)as in_f:
		lst=in_f.readlines()
		count={}
		for i in lst:
			#if i.split('\t')[10] not in count.keys():
			count[i.split('\t')[10]]=0							##########可以通过修改所在的列，统计激酶和底物的统计信息
	print 'all different sites:',len(count.keys())


##################统计出不同位点分别被多少个酶修饰###########################		
		for i in lst:
			if i.split('\t')[10] in count.keys():
				count[i.split('\t')[10]]+=1
				
#####################输出能被 指定数目激酶修饰 的位点 的个数####################
				specific_site=0
		for i in count.values():
			if i ==specify:
				specific_site+=1
		print 'the number of site which is phosphorylated by %d kinases is:' % specify,specific_site
		
		# max_value=0;max_key=0
		# for i in count.keys():
			# if count[i]>max_value:
				# max_value=count[i]
				# max_key=i
		# print max_key,max_value
		
		in_f.close()

###################找出所有 被特定数量激酶修饰的 位点，生成列表#####################
		multi=[]
		for i in count.keys():
			if count[i]==specify:
				multi.append(i)
				
####################对列表中的每一个位点，找出修饰他的多个激酶#####################				
		for k in multi:
			kinases=[]
			for x in lst:
				if x.split('\t')[10]==k:
					kinases.append( x.split('\t')[1])
			print kinases		

########################将激酶根据激酶分类文件进行分类##############################
			with open(filename2)as in_f:
				category=in_f.readlines();group=[]
				for x in kinases:
					for i in category:
						if len(re.findall(x,i .split(' ')[0]))!=0 and i.split(' ')[1] not in group:
								group.append(i.split(' ')[1])
				print group
				
			in_f.close()

	
if __name__=='__main__':

	open_multi(r'E:\python_project\data\multi.csv',r'E:\python_project\data\cate_kinase.csv',5)


