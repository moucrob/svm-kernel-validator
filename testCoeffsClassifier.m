clear all
X=[1 0.5 ;
   4 3.5 ;
   0.5 1 ;
   3.5 4];
y=[-1 ;
   -1 ;
    1 ;
    1];
Mdl = fitcsvm(X,y)
x=linspace(0,5,100);
figure ; plot(X(:,1), X(:,2), '.');
hold on ; plot(x,Mdl.Beta(2)*x + Mdl.Bias)