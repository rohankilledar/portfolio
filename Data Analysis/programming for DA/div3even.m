function div = div3even(n)
vec = randi([0,50],1,n);
evenVec = vec(2:2:length(vec));
index = ~mod(evenVec,3)==1;
div = evenVec(index);
end
