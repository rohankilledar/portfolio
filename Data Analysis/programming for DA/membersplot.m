% WRITE YOU CODE HERE
function membersplot(graph1,graph2)

%data=[ME 22;BM 45;CE 23;EE 33];
data = { "ME" (22) ;"BM" (45);"CE" (23);"EE" (33)};

%using two subplots with switch case to make the given graph type.
subplot(1,2,1)

switch graph1
    case 'bar'
     eval('bar([data{:,2}])')
     set(gca,'XtickLabel',data(:,1))
     xlabel('Departments')
     ylabel('Number of Faculty members')
     grid on
    
    case 'barh'
    eval('barh([data{:,2}])')
    set(gca,'YtickLabel',data(:,1))
    ylabel('Departments')
    xlabel('Number of Faculty members')
    grid on 

    case 'pie'
    eval('pie([data{:,2}],data(:,1))')
    title('Faculty members by department')
    otherwise
         warning('Unexpected plot type. No plot created.')
end   
   subplot(1,2,2)
   
   switch graph2
    case 'bar'
     eval('bar([data{:,2}])')
     set(gca,'XtickLabel',data(:,1))
     xlabel('Departments')
     ylabel('Number of Faculty members')
     grid on
    
    case 'barh'
    eval('barh([data{:,2}])')
    set(gca,'YtickLabel',data(:,1))
    ylabel('Departments')
    xlabel('Number of Faculty members')
    grid on

    case 'pie'
    eval('pie([data{:,2}],data(:,1))')
    title('Faculty members by department')
    otherwise
         warning('Unexpected plot type. No plot created.')
    
    end

end