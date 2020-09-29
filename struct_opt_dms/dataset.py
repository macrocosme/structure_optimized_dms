from .utils import load, save, run_fast_scandir
import numpy as np
import pandas as pd
from astropy.time import Time, TimeDelta

def load_dataframe(input_filterbanks_repository, state_variable_name, reload=False):
    if reload:
        print ('Reloading dataframe %s' % (state_variable_name))
        df_R3 = load(state_variable_name)
    else:
        print ('Setting dataframe %s from scratch' % (state_variable_name))
        zapped_channels = np.array([189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 1534, 1535])

        detection_folders = ['2020-03-22-10:03:39.R3', '2020-03-23-11:05:38.R3', '2020-03-23-11:05:38.R3', '2020-03-23-11:05:38.R3', '2020-05-09-11:45:55.R3', '2020-05-10-09:41:43.R3', '2020-05-11-07:36:22.R3', '2020-05-11-07:36:22.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-10:42:26.R3', '2020-05-11-14:40:00.R3', '2020-05-11-14:40:00.R3', '2020-05-11-14:40:00.R3', '2020-05-12-08:36:35.R3', '2020-05-27-03:20:38.R3', '2020-05-27-03:20:38.R3', '2020-05-27-03:20:38.R3', '2020-05-27-03:20:38.R3', '2020-05-27-07:21:12.R3', '2020-05-27-07:21:12.R3', '2020-05-27-10:52:06.R3', '2020-05-27-13:37:55.R3', '2020-05-27-13:37:55.R3', '2020-05-28-03:45:00.R3', '2020-05-28-05:13:48.R3', '2020-05-28-05:13:48.R3', '2020-05-28-08:19:28.R3', '2020-05-28-08:19:28.R3', ]
        observation_datetimes = ['%s%s%s' % (d[:10], 'T', d[11:].replace('.R3', '.0')) for d in detection_folders]
        detection_times = [4590.8, 4354.47, 7599.3, 9402.4, 9363.27, 9234, 3610.84, 6219.06, 4.15, 1780.7, 2317.14, 2688.72, 7153.32, 8913, 8959.9, 1495.37, 1889.1, 4387.14, 2216.21, 1612.21, 4810.98, 8867.2, 11658.7, 3592.32, 5082.87, 2616.07, 3437.12, 4390.28, 141.992, 2063.73, 4728.98, 1220.04, 3843.53, ]
        detection_mjd = Time(observation_datetimes, format='isot') + TimeDelta(detection_times, format='sec')
        detection_mjd = detection_mjd.mjd
        detection_dm = [348, 349, 348, 348, 348.2, 349, 348.2, 350.8, 352, 347.61, 349.2, 350.2, 354.45, 348, 350, 349.4, 348.06, 348.24, 350.3, 352.05, 349.49, 348.71, 349.2, 352.05, 354.2, 348.8, 351.8, 359.34, 348.4, 350.4, 349.56, 349.89, 349.09, ]
        detection_downsampling = [5.0, 5.0, 25.0, 5.0, 5.0, 10.0, 10.0, 25.0, 25.0, 100.0, 25.0, 10.0, 50.0, 1., 5.0, 10.0, 25.0, 10.0, 25.0, 50.0, 5.0, 5.0, 10.0, 50.0, 50.0, 5.0, 25.0, 250.0, 5.0, 25.0, 10.0, 5.0, 5.0]
        detection_snr = [11.5, 12.7, 13.4, 13.4, 13.6, 8.88, 16.38, 29.89, 13.86, 17.79, 10.12, 11.02, 38.61, 14, 12.5, 11.47, 58.14, 25.56, 31.5, 12.05, 12.67, 20.23, 20.95, 21.45, 19.48, 20.98, 20.87, 9.03, 25.71, 36.54, 29.9, 29.2, 20.92]
        detection_files, detection_filenames = [], []

        _, local_files = run_fast_scandir(input_filterbanks_repository, ['.fil'])
        arts_files = [
         '/tank/data/FRBs/R3/20200527/2020-05-27-10:52:06.R3/snippet/all/CB00_10.0sec_dm0_t02616_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-03:20:38.R3/snippet/all/CB00_10.0sec_dm0_t01612_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-03:20:38.R3/snippet/all/CB00_10.0sec_dm0_t011658_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-03:20:38.R3/snippet/all/CB00_10.0sec_dm0_t04810_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-03:20:38.R3/snippet/all/CB00_10.0sec_dm0_t08867_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-13:37:55.R3/snippet/all/CB00_10.0sec_dm0_t03437_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-13:37:55.R3/snippet/all/CB00_10.0sec_dm0_t04390_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-07:21:12.R3/snippet/all/CB00_10.0sec_dm0_t05082_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-07:21:12.R3/snippet/all/CB00_10.0sec_dm0_t03592_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200512/snippet/all/CB00_10.0sec_dm0_t02216_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200322/snippet/all/CB00_10.0sec_dm0_t04590_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200529/2020-05-29-03:20:12.R3/snippet/CB00_10.0sec_dm0_t01815_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t02317_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t03610_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t01495_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t01780_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t02688_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t06219_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t01889_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t08913_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t07153_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t04387_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t08959_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200511/snippet/all/CB00_10.0sec_dm0_t04_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200509/snippet/all/CB00_10.0sec_dm0_t09363_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200528/2020-05-28-03:45:00.R3/snippet/all/CB00_10.0sec_dm0_t0141_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200528/2020-05-28-08:19:28.R3/snippet/all/CB00_10.0sec_dm0_t03843_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200528/2020-05-28-08:19:28.R3/snippet/all/CB00_10.0sec_dm0_t01220_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200528/2020-05-28-05:13:48.R3/snippet/all/CB00_10.0sec_dm0_t04729_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200528/2020-05-28-05:13:48.R3/snippet/all/CB00_10.0sec_dm0_t02063_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200510/snippet/all/CB00_10.0sec_dm0_t09234_sb-1_tab00.fil',
         '/tank/data/FRBs/R3/20200323/snippet/CB00_10.0sec_dm0_t04354_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200323/snippet/CB00_10.0sec_dm0_t09402_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200323/snippet/CB00_10.0sec_dm0_t07599_sb35_tab00.fil',
         '/tank/data/FRBs/R3/20200527/2020-05-27-07:21:12.R3/snippet/all/CB00_10.0sec_dm0_t010773_sb-1_tab00.fil']

        # Construct detection_files and detection_filenames
        for folder, t in zip(detection_folders, detection_times):
            date = "".join(folder.split('-')[:3])
            found = False
            for f in arts_files:
                # If time and date somehwere in filename or in folder
                if (str(int(t)).zfill(2) in f or str(int(t)+1).zfill(2) in f) and date in f:
                    for ff in local_files:
                        if f.split('/')[-1] == ff.split('/')[-1]:
                            detection_files.append(ff)
                            detection_filenames.append(ff.split('/')[-1].split('.fil')[0])

                            found = True
                            break
                if found:
                    break
            if not found:
                detection_files.append('')
                detection_filenames.append('')

        df_R3 = pd.DataFrame({'detection_folder': detection_folders,
                              'observation_datetime': observation_datetimes,
                              'detection_time': detection_times,
                              'detection_mjd': detection_mjd,
                              'detection_dm': detection_dm,
                              'detection_downsampling': detection_downsampling,
                              'detection_snr': detection_snr,
                              'filename': detection_filenames,
                              'file_location': detection_files}).sort_values('detection_snr', ascending=False)

        df_R3['detection_isot'] = pd.to_datetime(df_R3['observation_datetime'], format='%Y-%m-%dT%H:%M:%S.%f') + pd.to_timedelta(df_R3['detection_time'], unit='s')

        phase0 = 58369.30
        t0 = Time(phase0, format='mjd')
        period = 16.28
        df_R3['detection_phase'] = df_R3['detection_mjd'].apply(lambda x: ((x - t0.mjd) % period) / period)

        df_R3 = df_R3.sort_values('detection_mjd', ascending=True)

        for i, row in df_R3.iterrows():
            df_R3.at[i, 'paper_name'] = "A" + str(i+1).zfill(2)

        save(state_variable_name, df_R3)
        df_R3.to_csv('arts_r3.csv', index=False)
    return df_R3
