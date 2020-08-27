openfemm;
main_maximize;
opendocument("motor2.FEM");
mi_modifycircprop("filed",1,1);
mi_modifycircprop("filed2",1,-1);
mi_modifycircprop("armature",1,10);
mi_modifycircprop("armature2",1,-10);
main_minimize;


n=1;
x=100;
for n = 1:x
    mi_minimize;
    mi_modifycircprop("armature",1,n/10);
    mi_modifycircprop("armature2",1,-n/10);
    mi_saveas("temp.FEM");
    mi_analyze(1)
    mi_loadsolution;
    mo_minimize; 
    mo_zoom(-10.45,-5.2,10.45,5.2);
    %mo_zoomnatural
    mo_smooth("on"); 
    mo_hidepoints; 


%    mo_hidedensityplot;
%    mo_savebitmap("current.bmp");
%    file = [int2str(n),"_1.bmp"];
%    rename("current.bmp",file);

%    mo_showdensityplot(1,0,0,0.1,"mag");
%    mo_savebitmap("current.bmp");
%    file = [int2str(n),".bmp"];
%    rename("current.bmp",file);
  
    mo_selectblock(3.5,0);
    mo_selectblock(-3.5,0);
  
    m(n)=mo_blockintegral(8);
    z(n)=mo_blockintegral(9);
    mo_selectblock(0,3.5);
    x(n)=mo_blockintegral(8);
    y(n)=mo_blockintegral(9);

end
mo_close();
mi_close();
n=0
closefemm;
%f = ["convert -delay 100 -loop 0 1.bmp"];
%for n = 2:x
%    g = [" ",int2str(n),".bmp"];
%    f = strcat(f,g);
%end
%f = strcat(f,[" output.gif"])
%system(f);

save field_x.csv m
save field_y.csv z

save air_x.csv x
save air_y.csv y
