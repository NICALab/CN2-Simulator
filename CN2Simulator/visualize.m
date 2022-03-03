%% parameters
height = 0.45;

%% draw entire spikes
figure(1);
NIDs = length(spike_time);
ntotal_spikes = 0;
for nid = 1:NIDs
    ntotal_spikes = ntotal_spikes + length(spike_time{nid});
end

xPoints = NaN(ntotal_spikes*3,1);
yPoints = xPoints;

idx = 1;
for nid = 1:NIDs
    spike_time_nid = spike_time{nid};
    nSpikes = length(spike_time_nid);
    tmp_xPoints = [spike_time_nid; spike_time_nid; NaN(1, nSpikes)];
    tmp_xPoints = tmp_xPoints(:);
    tmp_yPoints = [(nid - height)*ones(1, nSpikes);(nid + height)*ones(1, nSpikes); NaN(1, nSpikes)];
    tmp_yPoints = tmp_yPoints(:);
    xPoints(idx:idx+3*nSpikes-1) = tmp_xPoints;
    yPoints(idx:idx+3*nSpikes-1) = tmp_yPoints;
    idx = idx + 3*nSpikes;
end

plot(xPoints, yPoints, 'k');
ylim([0.5, NIDs + 0.5]);
xlim([min(xPoints)-0.5, max(xPoints)+0.5]);
hold on;

%% draw spike count rate
nid = 1;
windows = 5; % seconds
figure(2);
x_max = max(xPoints);
t_arr = windows:windows:x_max;
rate_arr = zeros(1, length(t_arr));
for idx = 1:length(t_arr)
    trange_e = t_arr(idx);
    trange_s = trange_e - windows;
    spike_time_nid = spike_time{nid};
    rate_arr(idx) = length(find(((spike_time_nid > trange_s) & (spike_time_nid < trange_e)) == 1))/windows;
end

plot(t_arr, rate_arr, "-o");

%% draw motif type1
figure(1);
motif_idx = 1;
NIDs = gt_type_1{motif_idx}.NIDs;
motif_times = gt_type_1{motif_idx}.motif_times;
for idx = 1:length(NIDs)
    nid = NIDs(idx) + 1;
    nSpikes = length(motif_times);
    tmp_xPoints = [motif_times;motif_times;NaN(1,nSpikes)];
    tmp_xPoints = tmp_xPoints(:);
    tmp_yPoints = [(double(nid)-height)*ones(1, nSpikes);(double(nid)+height)*ones(1, nSpikes);NaN(1,nSpikes)];
    tmp_yPoints = tmp_yPoints(:);
    plot(tmp_xPoints, tmp_yPoints, 'r')
end


%% draw motif type2
motif_idx = 1;
NIDs = gt_type_2{motif_idx}.NIDs;
lags = gt_type_2{motif_idx}.lags;
motif_times = gt_type_2{motif_idx}.motif_times;
for idx = 1:length(NIDs)
    nid = NIDs(idx) + 1;
    nSpikes = length(motif_times);
    tmp_xPoints = [motif_times+lags(idx);motif_times+lags(idx);NaN(1,nSpikes)];
    tmp_xPoints = tmp_xPoints(:);
    tmp_yPoints = [(double(nid)-height)*ones(1, nSpikes);(double(nid)+height)*ones(1, nSpikes);NaN(1,nSpikes)];
    tmp_yPoints = tmp_yPoints(:);
    plot(tmp_xPoints, tmp_yPoints, 'r')
end

%% draw motif type3
motif_idx = 1;
NIDs = gt_type_3{motif_idx}.NIDs;
lags = gt_type_3{motif_idx}.lags;
motif_times = gt_type_3{motif_idx}.motif_times;

%% draw motif type4
motif_idx = 1;
window_size = 0.3;
NIDs = gt_type_4{motif_idx}.NIDs;
motif_times = gt_type_4{motif_idx}.motif_times;
for idx = 1:length(NIDs)
    nid = NIDs(idx) + 1;
    nSpikes = length(motif_times);
    tmp_xPoints = [motif_times; motif_times + window_size; NaN(1,nSpikes)];
    tmp_xPoints = tmp_xPoints(:);
    tmp_yPoints = [double(nid)*ones(1, nSpikes); double(nid)*ones(1, nSpikes); NaN(1, nSpikes)];
    tmp_yPoints = tmp_yPoints(:);
    plot(tmp_xPoints, tmp_yPoints, 'r');
end

%% draw motif type5
motif_idx = 1;
window_size = 0.3;
NIDs = gt_type_5{motif_idx}.NIDs;
lags = gt_type_5{motif_idx}.lags;
motif_times = gt_type_5{motif_idx}.motif_times;
for idx = 1:length(NIDs)
    nid = NIDs(idx) + 1;
    nSpikes = length(motif_times);
    tmp_xPoints = [motif_times+lags(idx); motif_times + window_size+lags(idx); NaN(1,nSpikes)];
    tmp_xPoints = tmp_xPoints(:);
    tmp_yPoints = [double(nid)*ones(1, nSpikes); double(nid)*ones(1, nSpikes); NaN(1, nSpikes)];
    tmp_yPoints = tmp_yPoints(:);
    plot(tmp_xPoints, tmp_yPoints, 'r');
end

%% interspike interval histogram
isi = [];
for nid = 1:length(spike_time)
    a = spike_time{nid};
    for i = 1:length(a)-1
        isi = [isi, a(i+1)-a(i)];
    end
end
histogram(isi, 700);
