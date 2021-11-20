% WRITE HERE YOUR FUNCTION FOR EXERCISE 5
function answer = calctrianglearea(pmatrix)
plen= length(pmatrix);
trianglePoint = plen - rem(plen,3);
fprintf("The number of triangles that can be calculated using %d points are %d\n", plen, trianglePoint/3)

if rem(plen,3) ~=0
    fprintf("\nCoordinates that are not being used to construct triangles are: \n");
    disp(pmatrix(trianglePoint+1:end,:))
    else
        fprintf("\n All coordinates are used for making the triangles") ;
end

    pointa = pmatrix(1:3:trianglePoint,:);
    pointb = pmatrix(2:3:trianglePoint,:);       
    pointc = pmatrix(3:3:trianglePoint,:);

    ab = euclidean(pointa,pointb);
    bc = euclidean(pointb,pointc);
    ac = euclidean(pointa,pointc);

    s = (ab + bc + ac)./2;
    answer = sqrt(s.*(s-ab).* (s-bc).* (s-ac));

end

function sideLen =  euclidean(point1,point2)
    sideLen=sqrt((point1(:,1) - point2(:,1)).^2 + (point1(:,2) - point2(:,2)).^2);

 
end