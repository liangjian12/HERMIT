function [Y,Omega] = randmiss(Y_save,observe_rate)


 Omega = rand(size(Y_save))<observe_rate;
 

 Y =  Y_save.* Omega;
 

end