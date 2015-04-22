#!/bin/sh

#----------------------------------------------------
#文件存放形式为
#	dir/subdir1/files...
#	dir/subdir2/files...
#	dir/subdir3/files...
#	dir/subdirX/files...

#用法：
#1.$ sh mklabel.sh dir startlabel ;dir 为目标文件夹名称
#2.$ chmod a+x mklabel.sh ；然后可以直接用文件名运行
#3.默认label信息显示在终端，请使用转向符'>'生成文本，例：
#		$ sh ./mklabel.sh  data/faces94/male  > label.txt
#4.确保文件夹下除了图片不含其他文件(若含有则需自行添加判断语句)
#-----------------------------------------------------

DIR=/home/cari/caffe_150323/CarStyle_test/mklabel.sh		#命令位置（无用）
label=0					#label起始编号(为数字，根据自己需要修改)
testnum=0				#保留的测试集大小

if test $# -eq 0;then	#无参数，默认为当前文件夹下，label=1
	$DIR . 0 $label
else
	if test $# -eq 1;then	#仅有位置参数，默认testnum=0,label=1
		$DIR $1 0 $label
	else
		if test $# -eq 2;then	#两个参数时,label=1
			$DIR $1 $2 $label
		else
			testnum=$2			#每个类别保留测试集大小
			label=$3			#自定义label起始
			
			cd $1				#转到目标文件夹
		
			if test $testnum -ne 0;then
				mkdir "testdata"	#建立测试集
			fi
		
			for i in * ; do
				exist=`expr "$i" != "testdata"`
				if test -d $i && test $exist -eq 1;then	#文件夹存在
					#echo 
					#echo 'DIR:' $i
				
					cd $i			#进入文件夹
						num=1		#图片数目
						for j in *
						do
							if test $num -gt $testnum;then
								echo  $j  $label
								mv $j ../
							fi
							num=`expr $num + 1`
						done
					cd ..			#回到上层目录
				
					if test $testnum -eq 0;then
						rmdir $i
					else
						mv $i ./testdata
					fi
				
					label=`expr $label + 1`
									#计算label
				fi	
			done
		fi
	fi
fi
