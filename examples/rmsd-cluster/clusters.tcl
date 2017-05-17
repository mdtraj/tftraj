# 1
mol new ../../fs_peptide/fs-peptide.pdb
mol addfile cluster-0.nc waitfor all
animate delete beg 0 end 0 top
mol modstyle 0 top cpk 0.8 0.2
mol modmaterial 0 top Goodsell

mol addrep top
mol modstyle 1 top newcartoon
mol modcolor 1 top colorid 0
mol modmaterial 1 top Glossy

#2
mol new ../../fs_peptide/fs-peptide.pdb
mol addfile cluster-1.nc waitfor all
animate delete beg 0 end 0 top
mol modstyle 0 top cpk 0.8 0.2
mol modmaterial 0 top Goodsell

mol addrep top
mol modstyle 1 top newcartoon
mol modcolor 1 top colorid 1
mol modmaterial 1 top Glossy

color Name C white
source sscache.tcl
start_sscache
