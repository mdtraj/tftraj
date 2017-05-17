mol new ../../fs_peptide/fs-peptide.pdb
mol addfile consensus.nc waitfor all
animate delete beg 0 end 0 top
mol modstyle 0 top cpk 0.8 0.2
mol modmaterial goodsell

mol addrep top
mol modstyle 1 top newcartoon
mol modcolor 1 top colorid 1

color Name C white
source sscache.tcl
start_sscache
