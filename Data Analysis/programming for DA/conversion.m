function conv = conversion(measure, vector)

if measure == 'o' 
   conv = vector/0.035274;
   
 elseif measure == 'g' 
   conv = vector*0.035274;
     
else
    disp("Please provide measurement string as either 'o' or 'g' "); 
end
end

