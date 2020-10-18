% clear;

te_n_I = 1000;
te_n_T = 5000;
I_T = 5;
img_part_num = 36;
img_relation_num = 30;
embed_size = 1024;

% entity_mat_i2t = 'sim-small.mat';
% disp(['loading entity mat.....',entity_mat_i2t]);
% load(entity_mat_i2t);
% disp('loading entity mat completed.....');
% s_entity_i2t = similarity;

entity_mat_t2i = 'sim-large.mat';
disp(['loading entity mat.....',entity_mat_t2i]);
load(entity_mat_t2i);
disp('loading entity mat completed.....');
s_entity_t2i = similarity;

%%%whole%%%%
whole_mat = 'flickr_whole.mat';
disp(['loading whole mat.....',whole_mat]);
load(whole_mat);
disp('loading whole mat completed.....');
W_whole=test_txt*test_img';


% W=25*s_entity_i2t+25*s_entity_t2i+3*W_whole;

W=s_entity_t2i+W_whole;

x=[1,5,10];
[Y,ImgQuery] = sort(W',2,'descend');
for i=1:3
    R=x(i);
    res = ImgQuery(:,1:R);
    res_i_t = res;
    cnt = 0;
    for ii=1:te_n_I
        for j=1:R
            if I_T ==1
                if res(ii,j)==ii
                    cnt = cnt+1;
                    break;
                end
            else
                if (ii*5)>=res(ii,j)&&((ii-1)*5)<res(ii,j)
                    cnt = cnt+1;
                    break;
                end
            end
        end
    end
    disp(['R@' num2str(R) ':' num2str(cnt/te_n_I)]);
end

[Y,ImgQuery] = sort(W,2,'descend');
for i=1:3
    R=x(i);
    res = ImgQuery(:,1:R);
    res_t_i = res;
    cnt = 0;
    for ii=1:te_n_T
        for j=1:R
            if I_T ==1
                if res(ii,j)==ii
                    cnt = cnt+1;
                    break;
                end
            else
                if (res(ii,j)*5)>=ii&&((res(ii,j)-1)*5)<ii
                    cnt = cnt+1;
                    break;
                end
            end
        end
    end
    disp(['R@' num2str(R) ':' num2str(cnt/te_n_T)]);
end