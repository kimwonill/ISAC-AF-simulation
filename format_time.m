function s = format_time(t_sec)
% FORMAT_TIME  Pretty-print a duration in seconds as "Hh Mm Ss" / "Mm Ss" / "Ss".

if isnan(t_sec) || t_sec < 0
    s = '--';
    return;
end
h = floor(t_sec / 3600);
m = floor(mod(t_sec, 3600) / 60);
s_int = floor(mod(t_sec, 60));
if h > 0
    s = sprintf('%dh %02dm %02ds', h, m, s_int);
elseif m > 0
    s = sprintf('%dm %02ds', m, s_int);
else
    s = sprintf('%ds', s_int);
end
end
