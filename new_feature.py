#coding=utf-8


##本代码用于提取激酶的peptide上各位置的aa频率信息，上传到machine分支，删除该分支后再次创建分支并上传

import random
import pandas as pd

with open(r'C:\Users\xzw\Desktop\PKC_merge.fasta') as in_f:
	with open(r'C:\Users\xzw\Desktop\human_phosphoplus.csv') as in_f2:
#######################去除SS8蛋白质结构预测结果#################
		no_line3=[]
		for m,n in enumerate(in_f.readlines()):
			if (m-3)%6!=0:
				no_line3.append(n.strip('\n'))
		#print len(no_line3)
		
		
#################检查是否有蛋白质长度小于13##############
		b=0
		for i in no_line3:
			if '>' not in i and len(i)<=13:
				b+=1
		#print 0
		
#######################将各种预测信息分到不同字典中################		
		all_motif=[];c=0;d=0;e=0;f=0;g=0;h=0
		positive1={};positive2={};positive3={}
		all_dataset1={};all_dataset2={};all_dataset3={}
		for i in in_f2.readlines():
			if 'PKC' in i.split('\t')[1]:
				h+=1
			for m,n in enumerate(no_line3):
				if '|' in n:
					if 'PKC' in i.split('\t')[1] and i.split('\t')[6]==n.split('|')[1]:				#############原代码判断条件错误，没有加上CDK激酶这一限制条件
						g+=1
						for pos,aa in enumerate(no_line3[m+1]):
							if aa in ['Y','S','T']:
								if pos>5:
									if len(no_line3[m+1])-pos>6:
										if len(no_line3[m+1])-pos==7:
											all_dataset1[no_line3[m+1][pos-6:]]=no_line3[m+2][pos-6:]
											all_dataset2[no_line3[m+1][pos-6:]]=no_line3[m+3][pos-6:]
											all_dataset3[no_line3[m+1][pos-6:]]=no_line3[m+4][pos-6:]
											
										else:
											all_dataset1[no_line3[m+1][pos-6:pos+7]]=no_line3[m+2][pos-6:pos+7]
											all_dataset2[no_line3[m+1][pos-6:pos+7]]=no_line3[m+3][pos-6:pos+7]
											all_dataset3[no_line3[m+1][pos-6:pos+7]]=no_line3[m+4][pos-6:pos+7]
									else:
										all_dataset1[no_line3[m+1][pos-6:]+(pos+7-len(no_line3[m+1]))*'X']=no_line3[m+2][pos-6:]+(pos+7-len(no_line3[m+1]))*'X'	
										all_dataset2[no_line3[m+1][pos-6:]+(pos+7-len(no_line3[m+1]))*'X']=no_line3[m+3][pos-6:]+(pos+7-len(no_line3[m+1]))*'X'	
										all_dataset3[no_line3[m+1][pos-6:]+(pos+7-len(no_line3[m+1]))*'X']=no_line3[m+4][pos-6:]+(pos+7-len(no_line3[m+1]))*'X'	
								else:
									all_dataset1['X'*(6-pos)+no_line3[m+1][:pos+7]]='X'*(6-pos)+no_line3[m+2][:pos+7]
									all_dataset2['X'*(6-pos)+no_line3[m+1][:pos+7]]='X'*(6-pos)+no_line3[m+3][:pos+7]
									all_dataset3['X'*(6-pos)+no_line3[m+1][:pos+7]]='X'*(6-pos)+no_line3[m+4][:pos+7]

					
					
						postion=int(i.split('\t')[9][1:])
						if postion>6:
							if len(no_line3[m+1])-postion>5:
								if len(no_line3[m+1])-postion==6:
									c+=1																						#找出motif刚好在蛋白质末端的数量
									positive1[no_line3[m+1][postion-7:]]=no_line3[m+2][postion-7:]
									positive2[no_line3[m+1][postion-7:]]=no_line3[m+3][postion-7:]
									positive3[no_line3[m+1][postion-7:]]=no_line3[m+4][postion-7:]
									# if len(n[int(i.split('\t')[9][1:]-7:])!=13:
										# print n[int(i.split('\t')[9][1:]-7:]
									if no_line3[m+1][postion-7:] not in all_motif:														#添加无重复的motif
										all_motif.append(no_line3[m+1][postion-7:])
								else:
									d+=1																							#找出motif在蛋白质中间的数量
									positive1[no_line3[m+1][postion-7:postion+6]]=no_line3[m+2][postion-7:postion+6]
									positive2[no_line3[m+1][postion-7:postion+6]]=no_line3[m+3][postion-7:postion+6]
									positive3[no_line3[m+1][postion-7:postion+6]]=no_line3[m+4][postion-7:postion+6]
									# if len(n[int(i.split('\t')[9][1:]-7:int(i.split('\t')[9][1:]+6])!=13:
										# print n[int(i.split('\t')[9][1:]-7:int(i.split('\t')[9][1:]+6]
									if no_line3[m+1][postion-7:postion+6] not in all_motif:											#添加无重复的motif
										all_motif.append(no_line3[m+1][postion-7:postion+6])
							else:
								e+=1																								#找出在蛋白质末端且需要添加X的motif的数量
								positive1[no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X']=no_line3[m+2][postion-7:]+(postion+6-len(no_line3[m+1]))*'X'			#为什么+9才能正常运行？
								positive2[no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X']=no_line3[m+3][postion-7:]+(postion+6-len(no_line3[m+1]))*'X'
								positive3[no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X']=no_line3[m+4][postion-7:]+(postion+6-len(no_line3[m+1]))*'X'
								if len(no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X')!=13:
									print no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X'											#输出异常数据
									
								if no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X' not in all_motif:									#添加无重复的motif
									all_motif.append(no_line3[m+1][postion-7:]+(postion+6-len(no_line3[m+1]))*'X')
						else:
							f+=1																									#找出在蛋白质前端（添加X）的motif的数量
							
							positive1['X'*(7-postion)+no_line3[m+1][:postion+6]]='X'*(7-postion)+no_line3[m+2][:postion+6]
							positive2['X'*(7-postion)+no_line3[m+1][:postion+6]]='X'*(7-postion)+no_line3[m+3][:postion+6]
							positive3['X'*(7-postion)+no_line3[m+1][:postion+6]]='X'*(7-postion)+no_line3[m+4][:postion+6]
							# if len('X'*(7-postion+n[:int(i.split('\t')[9][1:]+6])!=13:
								# print 'X'*(7-postion+n[:int(i.split('\t')[9][1:]+6]
							if 'X'*(7-postion)+no_line3[m+1][:postion+6] not in all_motif:
								all_motif.append('X'*(7-postion)+no_line3[m+1][:postion+6])
								
		print g,h	########源文件中某个激酶对应的底物的所有条目			
#################鉴定是否存在异常数据########################
		error=0
		for i in positive1.keys():
			if len(i)!=13:
				error+=1
				#print i,all_dataset1[i]
				del positive1[i]
		for i in positive2.keys():
			if len(i)!=13:
				del positive2[i]
		for i in positive3.keys():
			if len(i)!=13:
				del positive3[i]
		print 'the number of abnormal data:',error
		print 'all identified S/Y/T sites:',len(positive1.keys());print 'all (un)identified S/Y/T sites:',len(all_dataset3.values())	#结果有17303个motif，但在原始文件中存在17733个位点，因此应该有430个重复位点
		print len(all_motif)			##############所有无重复的磷酸化motif

		final_vec1 = []
		T_positive = [acid for acid in positive1.keys() if acid[6] in ['T', 'Y', 'S']]
		T_positive2 = [i for i in T_positive if 'X' not in i]
		final_vec2 = []
		T_all = [acid for acid in all_dataset1.keys() if acid[6] in ['T', 'Y', 'S']]
		T_all_negative = [i for i in T_all if i not in T_positive]
		T_all_negative2 = [i for i in T_all_negative if 'X' not in i]
		T_negative2 = random.sample(T_all_negative2, len(T_positive2))
		all_dataset = T_negative2 + T_positive2
		print len(T_negative2)
		
############在CDK激酶的底物中，存在一个不是SYT的磷酸化位点：CMGGMNRrPILTIIT；另外，P68431底物的S11位点被磷酸化，但原文件统计时，写成了s10	
		# T_abnormal=[i for i in positive1.keys() if i[6] not in ['T','S','Y']]
		# print T_abnormal
		
		
		# T_normal=[i for i in positive1.keys() if i[6] in ['T','S','Y']]
		# print len(T_normal)
		
		
		CDK_frequency=[]
		for x in range(13):
			frequency={'C':0,'S':0,'T':0,'P':0,'A':0,'G':0,'N':0,'D':0,'E':0,'Q':0,'H':0,'R':0,'K':0,'M':0,'I':0,'L':0,'V':0,'F':0,'Y':0,'W':0}
			for i in T_positive2:
				if i[x] in frequency.keys():
					frequency[i[x]]+=1
					
			CDK_frequency.append(frequency)
		
		print CDK_frequency
		
		
	in_f2.close()
in_f.close()	
		
freq=pd.DataFrame(CDK_frequency)

freq['sum']=freq.T.loc[:,1].sum()
freq=freq/freq.loc[1,'sum']
print freq.T
		
freq.T.to_excel(r'PKC_frequency.xls')
		
		
				