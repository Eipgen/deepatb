
cd file;
ls *aip | awk '{print "amesp "$1}' | sh
# you need to check the aop file success or not;
rm *.mo
cd ../
grep "Final Energy" */*aop >atb_ene.txt
