clc
clear
close all

p='C:\Users\ericj\Downloads\epxtmp_0001\dif-cal-hrii_hriv_mri-6-epoxi-temps-v2.0\data\smooth\hri';

dinfo=dir(p);
files={dinfo.name};
files=files(contains(files,'.txt'));
files=files(contains(files,'_i_'));
fin=[];
for i=1:length(files)
    f=[p '\' files{i}];
    data=readtimetable(f);
    fin=[fin;data];
end

% fin2=unique(sortrows(fin));
% fin=fin2;
p='C:\Users\ericj\Downloads\epxtmp_0001\dif-cal-hrii_hriv_mri-6-epoxi-temps-v2.0\data\smooth\mri';

dinfo=dir(p);
files={dinfo.name};
files=files(contains(files,'.txt'));
files=files(contains(files,'_i_'));
finm=[];
for i=1:length(files)
    f=[p '\' files{i}];
    data=readtimetable(f);
          finm=[finm;data];
end


vr=fin.Properties.VariableNames(contains(fin.Properties.VariableNames,'I_'));
vr2=fin.Properties.VariableNames(contains(fin.Properties.VariableNames,'CI'));
fin=removevars(fin,vr2);
vr2=fin.Properties.VariableNames(contains(fin.Properties.VariableNames,'JD'));
fin=removevars(fin,vr2);

for i=1:length(vr)
    fin(find(fin.([vr{i}])<-400),:)=[];
end
fin=retime(fin,'hourly','linear');

vr=finm.Properties.VariableNames(contains(finm.Properties.VariableNames,'I_'));
vr2=finm.Properties.VariableNames(contains(finm.Properties.VariableNames,'CI'));
finm=removevars(finm,vr2);
vr2=finm.Properties.VariableNames(contains(finm.Properties.VariableNames,'JD'));
finm=removevars(finm,vr2);

for i=1:length(vr)
    finm(find(finm.([vr{i}])<-400),:)=[];
end
finm=retime(finm,'hourly','linear');

figure

% TT=retime(fin,'hourly','linear');

fin=[fin finm];

vr=fin.Properties.VariableNames(contains(fin.Properties.VariableNames,'I_'));
for i=1:length(vr)
    %     fin(find(fin.([vr{i}])<-400),:)=[];
    plot(fin.SCET,fin.([vr{i}]))
    hold on

end
legend(vr,'interpreter','none')

writetimetable(fin,'./test.csv')

