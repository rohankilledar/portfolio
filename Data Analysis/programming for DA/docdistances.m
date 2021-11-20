% WRITE YOU CODE HERE

q = char(39);

corpus = cell(0);
nofDocsInCorpus = 6;
for i = 1:nofDocsInCorpus
% opening each file one at a time in the loop
if (i==1)
    fileID = 'RedRidingHood.txt';
    
end

if(i==2)
    fileID = 'PrincessPea.txt';
end

if(i==3)
    fileID = 'Cinderella.txt';
end

if(i==4)
    fileID = 'CAFA1.txt';
end

if(i==5)
fileID = 'CAFA2.txt';
end

if(i==6)
fileID = 'CAFA3.txt';
end

%stringy =strcat('textscan(fileID',num2str(i),',',q,'%s',q,')');
file = fileread(fileID);
stri = lower(strtrim(file));

%getting all the lower case text in doc1,doc2,...doc6 as we will need it
%later to find if the term is there in the respective docs
doc = strcat('doc',num2str(i),'= stri;');
eval(doc);

op= split(stri," ");
%stringy =strcat('textscan(fileID',num2str(i),',',q,'Delimiter',q,',','" ")');
%op = eval(stringy);
st = lower(op);

%using unique to find the different terms and their occurance in the given
%document
[wo,~,ic] = unique(st);

%occurance = accumarray(ic,1);

%%initialize the tf 
%stringtf = strcat('tf',num2str(i),'=zeros(',num2str(length(wo)),',1)');
%eval(stringtf)



%concat the terms obtained to corpus
corpus = [corpus;wo];
%close any file currently open
fclose('all');
end

%to find the Term frequency for all the documents:
for i = 1:nofDocsInCorpus
stringtf = strcat('tf',num2str(i),'=zeros(',num2str(length(corpus)),',1);');

%initialized tf1,tf2,...,tf6 for each document to the length of the
%corpus
eval(stringtf);

for j = 1:length(corpus)
    %getting the count of each term in doc(i) for each term in the corpus
    %and assigning the value to tf(i).
    occuranceString = strcat('count(doc',num2str(i),',corpus{',num2str(j),'})');
    occurance = eval(occuranceString);
    vecString = strcat('tf',num2str(i),'(',num2str(j),') = occurance;');
    eval(vecString);
end

end

%to find inverse document frequency, using the same logic as tf.
for i = 1:nofDocsInCorpus
    %initializing all the idf
stringtf = strcat('idf',num2str(i),'=zeros(',num2str(length(corpus)),',1);');
eval(stringtf);

for j = 1:length(corpus)
    %creating a counter for in how many document the term appears.
    nDocsTA = 0;
    for k = 1:nofDocsInCorpus
       checkcond = strcat('if tf',num2str(k),'(',num2str(j),') nDocsTA=nDocsTA+1; end');
       eval(checkcond);
    end
        
    idf = log10(nofDocsInCorpus/nDocsTA);
    
    vecString = strcat('idf',num2str(i),'(',num2str(j),') = idf;');
    eval(vecString);
end

end

%finding the document vector tf-idf for all documents
for i = 1:nofDocsInCorpus
    %calculating tf-idf
    dvString = strcat('dv',num2str(i),'= tf',num2str(i),'.*idf',num2str(i),';');
    eval(dvString);
    
    %finding the norm of each document vector tf-idf
    normString = strcat('norm',num2str(i),'= norm(','dv',num2str(i),');');
    eval(normString);
    
    %dividing tf-idf by the norm calculated that will be needed to find the
    %cos similarity
    vdString = strcat('vd',num2str(i),'= dv',num2str(i),'/norm',num2str(i),';');
    eval(vdString);
end

cosineMat = zeros(nofDocsInCorpus,nofDocsInCorpus);
%finding the cosine distance between all pair of documents and putting it
%in a matrix 
for i = 1:nofDocsInCorpus
for j = 1:nofDocsInCorpus
     cosMatString = strcat('cosineMat(',num2str(i),',',num2str(j),')= 1 - dot(vd',num2str(i),',vd',num2str(j),');');
     eval(cosMatString);
end
end

%displaying the matrix using imagesc.
imagesc(cosineMat)
colorbar
colormap gray
set(gca,'XTickLabel',["RRH","PPea","Cinde","CAFA1","CAFA2","CAFA3"])
set(gca,'YTickLabel',["RRH","PPea","Cinde","CAFA1","CAFA2","CAFA3"])