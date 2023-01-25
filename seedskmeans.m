%% clear variables, close everything and load data
clear
close all

%load entire dataset, including labels
seeds_dataset_complete = load('seeds_dataset_no_table.txt');
height_dataset_complete = size(seeds_dataset_complete, 1);
depth_dataset_complete = size(seeds_dataset_complete, 2);

%create dataset without labels
seeds_dataset = seeds_dataset_complete(:,1:depth_dataset_complete-1);
height_dataset = size(seeds_dataset, 1);
depth_dataset = size(seeds_dataset, 2);

DataCols=["Area", "Perimeter", "Compactness", "L of kernel", "W of kernel", "Asymmetry", "L of kernel groove"];

%number of possible 2D dataspaces given the dimension of each instance
ncombinations = nchoosek(size(seeds_dataset, 2), 2);

%normalization between 0 and 1 of the dataset
norm_dataset = seeds_dataset;
for i = 1:depth_dataset
    max_val = max(norm_dataset(:,i));
    norm_dataset(:,i) = norm_dataset(:,i)/max_val;
end

%% Accuracy of K-means for entire dataset

[idx, C, sumd, D] = kmeans(norm_dataset,3);

accuracy_entire = accuracy(idx); 
%Function "accuracy(x)" determines the proportion of data points that have 
%been correctly labeled.

%% Scree plot all dimensions of dataset using Kmeans++, 2 to 30 clusters


tic
k=2:30;
nk=length(k);
distortion_array = [];
for i=1:nk
k(i);
[Cluster_IDs_plus,centroids_plus,distortion_plus]=kmeans(norm_dataset,k(i), "Display","iter");
distortion_array =[distortion_array, sum(distortion_plus)];
end
kmeansplus30time = toc; %measuring time of entire process

figure(1)
plot(k,distortion_array,'b-*')
title('Scree plot of K-means with K-means++ initialization', 'Interpreter','latex','FontSize',14);
xlabel("Number of clusters",'Interpreter','latex','FontSize',14);
ylabel("Distortion measure",'Interpreter','latex','FontSize',14);
str1 = sprintf('%i',kmeansplus30time);
str2 = strcat("time = ",str1, " s");
t = text(0.55,0.9,str2,'Units','normalized');
t2 = text(0.55,0.82,"avg. iterations = 6.85",'Units','normalized');
t.FontSize = 12;
t2.FontSize = 12;

%The number of iterations is displayed on the console and manually inserted
%in an Excel Sheet

%% Scree plot of all dimensions of dataset using random sample initialization, 2 to 30 clusters

tic
k=2:30;
nk=length(k);
distortion_array = [];
for i=1:nk
i+1;
[Cluster_IDs_rand,centroids_rand,distortion_rand]=kmeans(norm_dataset,k(i),"Start", "sample", "Display","iter");
distortion_array =[distortion_array, sum(distortion_rand)];
end
kmeansrand30time = toc;

figure(2)
plot(k,distortion_array,'b-*')
title('Scree plot of K-means with random sample initialization', 'Interpreter','latex','FontSize',14);
xlabel("Number of clusters",'Interpreter','latex','FontSize',14);
ylabel("Distortion measure",'Interpreter','latex','FontSize',14);
str1 = sprintf('%i',kmeansrand30time);
str2 = strcat("time = ",str1, " s");
t = text(0.55,0.9,str2,'Units','normalized');
t2 = text(0.55,0.82,"avg. iterations = 9.01",'Units','normalized');
t.FontSize = 12;
t2.FontSize = 12;

%% K-means and plots of all possible 2D dataspaces to check which combinations have better results

accuracy_array = zeros(ncombinations,1);
ids_array = zeros(height_dataset,ncombinations);

the_colors=colormap('lines');

%figure containing all the possible subplots
figure(3)
k = 0;
for i = 1:depth_dataset
    i;
    for j = (i+1):depth_dataset
        j;
        [ids, centroids_all] = kmeans([seeds_dataset(:,i), seeds_dataset(:,j)],3);
        accuracy_percent = round(accuracy(ids),2);
        k = k + 1;
        accuracy_array(k,1) = accuracy_percent;
        ids_array(:,k) = ids;
        subplot(depth_dataset, 3, k)
        scatter(seeds_dataset(:,i), seeds_dataset(:,j),10,the_colors(ids,:),'fill');
        hold on
        voronoi(centroids_all(:,1),centroids_all(:,2));
        scatter(centroids_all(:,1),centroids_all(:,2),30,'^r','fill');
        labelList={DataCols(i), DataCols(j)};
        xlabel(labelList{1}+newline+"   ",'Interpreter','latex');
        ylabel(labelList{2}+newline+"   ",'Interpreter','latex');
        text(0.05,0.8,sprintf('%.2f',accuracy_percent),'Units','normalized')
        t.FontSize = 12;
        hold off
    end
end
sgtitle('Clustering of each possible 2D data space combination', 'Interpreter','latex','FontSize',14);

%Separation of all the subplots into three different figures
k = 0;
for i = 1:depth_dataset
    i;
    for j = (i+1):depth_dataset
        j;
        [ids, centroids_all] = kmeans([norm_dataset(:,i), norm_dataset(:,j)],3);
        accuracy_percent = round(accuracy(ids),2);
        k = k + 1;

        if k < 10
            figure(4)
            subplot(3, 3, k)
            scatter(norm_dataset(:,i), norm_dataset(:,j),10,the_colors(ids,:),'fill');
            hold on
            voronoi(centroids_all(:,1),centroids_all(:,2));
            scatter(centroids_all(:,1),centroids_all(:,2),30,'^r','fill');
            %scatter(norm_dataset(:,i), norm_dataset(:,j), 5, "*")
            labelList={DataCols(i), DataCols(j)};
            xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
            ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
            text(0.05,0.8,sprintf('%.2f',accuracy_percent),'Units','normalized')
            t.FontSize = 12;
            hold off
            sgtitle('Clustering of each possible 2D data space combination, first 9 combinations', 'Interpreter','latex','FontSize',14);

        elseif k > 9 && k<19
            figure(5)
            subplot(3, 3, k-9)
            scatter(norm_dataset(:,i), norm_dataset(:,j),10,the_colors(ids,:),'fill');
            hold on
            voronoi(centroids_all(:,1),centroids_all(:,2));
            scatter(centroids_all(:,1),centroids_all(:,2),30,'^r','fill');
            labelList={DataCols(i), DataCols(j)};
            xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
            ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
            text(0.05,0.8,sprintf('%.2f',accuracy_percent),'Units','normalized')
            t.FontSize = 12;
            hold off
            sgtitle('Clustering of each possible 2D data space combination, combinations 10-18', 'Interpreter','latex','FontSize',14);

        else
            figure(6)
            subplot(1, 3, k-18)
            scatter(norm_dataset(:,i), norm_dataset(:,j),10,the_colors(ids,:),'fill');
            hold on
            voronoi(centroids_all(:,1),centroids_all(:,2));
            scatter(centroids_all(:,1),centroids_all(:,2),30,'^r','fill');
            labelList={DataCols(i), DataCols(j)};
            xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
            ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
            text(0.05,0.8,sprintf('%.2f',accuracy_percent),'Units','normalized')
            t.FontSize = 12;
            hold off
            sgtitle('Clustering of each possible 2D data space combination, last 3 combinations', 'Interpreter','latex','FontSize',14);
        end
    end
end


%% Scatterplot of interesting dataset overlayed with correct results

%area and asymmetry chosen as interesting 2D dataset
X_vals = norm_dataset(:,[1 6]);
labelList={DataCols(1), DataCols(6)};

the_colors=colormap('lines');

[Cluster_IDs_plus,centroids_plus,distortion_plus]=kmeans(X_vals,3, "Display","iter");

%alters order of colors, rerun this entire section if overlay colors don't
%generally match
for i = 1:height_dataset_complete

    if Cluster_IDs_plus(i) == 2
        
        Cluster_IDs_plus(i) = 3;

    elseif Cluster_IDs_plus(i) == 3
            
        Cluster_IDs_plus(i) = 2;
    end


end

%plot of data as clustered by k-means
figure(7)
scatter(X_vals(:,1), X_vals(:,2),30,the_colors(Cluster_IDs_plus,:),'fill');
hold on
voronoi(centroids_plus(:,1),centroids_plus(:,2));
scatter(centroids_plus(:,1),centroids_plus(:,2),50,'^r','fill');
title('Data as clustered by K-means with K-means++ init.', 'Interpreter','latex','FontSize',14);
xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
hold off

%plot of data as labeled
figure(8)
centroids_correct = [mean(X_vals(1:70,1)), mean(X_vals(1:70,2)); mean(X_vals(70:140,1)), mean(X_vals(70:140,2)); mean(X_vals(140:210,1)), mean(X_vals(140:210,2))]
scatter(X_vals(:,1), X_vals(:,2),45,the_colors(seeds_dataset_complete(:,8),:),'^');
hold on
voronoi(centroids_correct(:,1),centroids_correct(:,2), 'g');
scatter(centroids_correct(:,1),centroids_correct(:,2),50,'^g','fill');
title('Data clusters as given by labels', 'Interpreter','latex','FontSize',14);
xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
hold off

%overlay of the previous two plots
figure(9)
scatter(X_vals(:,1), X_vals(:,2),30,the_colors(Cluster_IDs_plus,:),'fill');
hold on
voronoi(centroids_plus(:,1),centroids_plus(:,2));
scatter(centroids_plus(:,1),centroids_plus(:,2),50,'^r','fill');
hold on
centroids_correct = [mean(X_vals(1:70,1)), mean(X_vals(1:70,2)); mean(X_vals(70:140,1)), mean(X_vals(70:140,2)); mean(X_vals(140:210,1)), mean(X_vals(140:210,2))]
scatter(X_vals(:,1), X_vals(:,2),45,the_colors(seeds_dataset_complete(:,8),:),'^');
hold on
voronoi(centroids_correct(:,1),centroids_correct(:,2), 'g');
scatter(centroids_correct(:,1),centroids_correct(:,2),50,'^g','fill');
title('Overlay of labeled and K-means++ data clustering', 'Interpreter','latex','FontSize',14);
xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
hold off



%% clustering interesting data space with 3 to 8 clusters using K-means++ initizalization

%area and asymmetry are the chosen 2D dataspace
X_vals = norm_dataset(:,[1 6]);
labelList={DataCols(1), DataCols(6)};

%scatter plots
tic
distortion_array = [];
the_colors=colormap('lines');
k=3:8;
figure(10)
nk=length(k);
for i=1:nk
i;
[Cluster_IDs_plus,centroids_plus,distortion_plus]=kmeans(X_vals,k(i), "Display","iter");
distortion_array =[distortion_array, sum(distortion_plus)];
nkk = nk/2;
subplot(2,nkk,i)
scatter(X_vals(:,1), X_vals(:,2),5,the_colors(Cluster_IDs_plus,:),'fill');
hold on
    if k(i)>2%voronoi function works only with at least 3 points
    voronoi(centroids_plus(:,1),centroids_plus(:,2));
    else
        unit_vector=(centroids_plus(1,:)-centroids_plus(2,:))/norm(centroids_plus(1,:)-centroids_plus(2,:));
        mid_point=(centroids_plus(1,:)+centroids_plus(2,:))/2;
        boundary_2clusters=[mid_point+[unit_vector(2),-unit_vector(1)]*-100;mid_point+[unit_vector(2),-unit_vector(1)]*100];
        plot(boundary_2clusters(:,1),boundary_2clusters(:,2));
    end
scatter(centroids_plus(:,1),centroids_plus(:,2),30,'^r','fill');
xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
str1 = sprintf('%i',k(i));
str2 = strcat("k = ",str1);
t = text(0.05,0.9,str2,'Units','normalized');
t.FontSize = 12;
hold off
end
sgtitle('Clustering interesting 2D data space with 3 to 8 clusters using K-means++ initizalization', 'Interpreter','latex','FontSize',14);
kmeansplusplus_time = toc

%scree plot
figure(11)
plot(k,distortion_array,'b-*')
title('Scree plot of K-means with K-means++ initialization, 2D data space', 'Interpreter','latex','FontSize',14)
xlabel("Number of clusters",'Interpreter','latex','FontSize',14);
ylabel("Distortion measure",'Interpreter','latex','FontSize',14);



%% Clustering interesting data space with 3 to 8 clusters using random sample initialization, 2 to 8 clusters


%area and asymmetry are the chosen dataspace
X_vals = norm_dataset(:,[1 6]);
labelList={DataCols(1), DataCols(6)};

%scatter plots
tic
distortion_array = [];
the_colors=colormap('lines');
k=3:8;
figure(12)
nk=length(k);
for i=1:nk
i;
[Cluster_IDs_rand,centroids_rand,distortion_rand]=kmeans(X_vals,k(i),"Start", "sample", "Display","iter");
distortion_array =[distortion_array, sum(distortion_rand)];
nkk = nk/2;
subplot(2,nkk,i)
scatter(X_vals(:,1), X_vals(:,2),5,the_colors(Cluster_IDs_rand,:),'fill');
hold on
    if k(i)>2%voronoi function works only with at least 3 points
    voronoi(centroids_rand(:,1),centroids_rand(:,2));
    else
        unit_vector=(centroids_rand(1,:)-centroids_rand(2,:))/norm(centroids_rand(1,:)-centroids_rand(2,:));
        mid_point=(centroids_rand(1,:)+centroids_rand(2,:))/2;
        boundary_2clusters=[mid_point+[unit_vector(2),-unit_vector(1)]*-100;mid_point+[unit_vector(2),-unit_vector(1)]*100];
        plot(boundary_2clusters(:,1),boundary_2clusters(:,2));
    end
scatter(centroids_rand(:,1),centroids_rand(:,2),30,'^r','fill');
xlabel(labelList{1}+newline+"   ",'Interpreter','latex','FontSize',14);
ylabel(labelList{2}+newline+"   ",'Interpreter','latex','FontSize',14);
str1 = sprintf('%i',k(i));
str2 = strcat("k = ",str1);
t = text(0.05,0.9,str2,'Units','normalized');
t.FontSize = 12;
hold off
end
sgtitle('Clustering interesting 2D data space with 3 to 8 clusters, random sample init.', 'Interpreter','latex','FontSize',14);
kmeansrand_time = toc

%scree plot
figure(13)
plot(k,distortion_array,'b-*')
title('Scree plot of K-means with sample initialization, 2D data space', 'Interpreter','latex','FontSize',14)
xlabel("Number of clusters",'Interpreter','latex','FontSize',14);
ylabel("Distortion measure",'Interpreter','latex','FontSize',14);

