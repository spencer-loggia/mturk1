##################################
# MTurk1 Decoding Power Analysis #
#     Helen Feibes 2023-08-31    #
##################################

library(dplyr)
library(tidyr)
library(ggplot2)

# For POA
# Same as below but includes a summarising step

# Function to wrangle dataset into necessary format for plotting and then plot it
for_plot <- function(train_set){
  # Prep entries for a type_decode column using name of train set
  if (deparse(substitute(train_set)) == "uncolored") {
    cross_name = "shape-to-color" 
    train_name = "shape-to-shape"
    plot_title = "Decoder trained on shape"
  } else{
    cross_name = "color-to-shape" 
    train_name = "color-to-color"
    plot_title = "Decoder trained on color"
  }
  plot_set <- train_set %>% 
    # Expand lengthwise in order to separate out cross and within decoding
    pivot_longer(cols = c(cross_mean, cross_var, train_mean, train_var),
                 names_to = c("type_decode", "stat"),
                 values_to = "value",
                 names_sep = "_") %>% 
    # Compress again
    pivot_wider(names_from = stat, values_from = value) %>% 
    mutate(type_decode = case_when(type_decode == "cross" ~ cross_name,
                                   type_decode == "train" ~ train_name)) 
  
  decode_plot <- ggplot(plot_set, aes(x=roi, y=mean, fill = type_decode)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7)+
    geom_linerange(aes(ymin=mean-var, ymax=mean+var), position = position_dodge(.7)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    labs(title = plot_title,
         y = "Decoding accuracy") +    
    scale_fill_npg() +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
}

# what datasets do you want to plot?
# what do you want to call them?


# This script is chaos right now.

jeeves_sc <- read.csv('/media/ssbeast/DATA/Users/Helen/20230903_200bfs_sc.csv') # 200 bfs
jeeves_cs <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_50bfs_cs.csv') # 50 bfs
#jeeves_cs_cbrs <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_50bfs_cs_antCBRs.csv')
#jeeves_sc_cbrs <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_50bfs_sc_antCBRs.csv')
jeeves_ss <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_jeeves_rh_50bfs_ss.csv')
jeeves_cc <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_jeeves_rh_50bfs_cc.csv')
aghhh <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_jeeves_rh_1000bfs_sc.csv')
jeeves_sc_pcaontest <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_jeeves_50bfs_sroi_sc_pcaontest.csv')


pcaontest <- jeeves_sc_pcaontest %>% 
  # pivot everything long
  pivot_longer(cols = c(cross_mean, cross_var, train_mean, train_var),
               names_to = c("idorcross", "meanorvar"),
               values_to = "value",
               names_sep = "_") %>%
  # pivot back
  pivot_wider(names_from = meanorvar, values_from = value)
(pcaontest_plot <- ggplot(pcaontest, aes(x=roi, y=mean, fill = idorcross)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7)+
    geom_linerange(aes(ymin=mean-var, ymax=mean+var), position = position_dodge(.7)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)


sad_overfitting <- aghhh %>% 
  # pivot everything long
  pivot_longer(cols = c(cross_mean, cross_var, train_mean, train_var),
               names_to = c("idorcross", "meanorvar"),
               values_to = "value",
               names_sep = "_") %>%
  # pivot back
  pivot_wider(names_from = meanorvar, values_from = value)
(s_o_plot <- ggplot(sad_overfitting, aes(x=roi, y=mean, fill = idorcross)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7)+
    geom_linerange(aes(ymin=mean-var, ymax=mean+var), position = position_dodge(.7)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)


cs <- jeeves_cs %>% 
  mutate(type_decode = "cs")
cc <- jeeves_cc %>% 
  mutate(type_decode = "cc")
sc <- jeeves_sc %>% 
  mutate(type_decode = "sc")
ss <- jeeves_ss %>% 
  mutate(type_decode = "ss")

train_shapes <- sc %>% 
  rbind(ss)

(train_shapes_plot <- ggplot(train_shapes, aes(x=roi, y=cross_mean, fill = type_decode)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7)+
    geom_linerange(aes(ymin=cross_mean-cross_var, ymax=cross_mean+cross_var), position = position_dodge(.7)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    labs(title = "train on shape, 50 bootfolds, last two TRs") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)
ggsave('/media/ssbeast/DATA/Users/Helen/train_shapes.png')

train_colors <- cs %>% 
  rbind(cc)
(train_colors_plot <- ggplot(train_colors, aes(x=roi, y=cross_mean, fill = type_decode)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7)+
    geom_linerange(aes(ymin=cross_mean-cross_var, ymax=cross_mean+cross_var), position = position_dodge(.7)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    labs(title = "train on color, 50 bootfolds, last two TRs") +
    theme_classic() +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
)
ggsave('/media/ssbeast/DATA/Users/Helen/train_colors.png')




# shape to color
shapes_sc <- jeeves_sc %>% 
  filter(grepl('shape_patch', roi))

places_sc <- jeeves_sc %>% 
  filter(grepl('P_jeeves', roi))

faces_sc <- jeeves_sc %>% 
  filter(grepl('face_patch', roi))

cbrs_sc <- jeeves_sc %>% 
  filter(grepl('CBR', roi))

# color to shape
shapes_cs <- jeeves_cs %>% 
  filter(grepl('shape_patch', roi))

places_cs <- jeeves_cs %>% 
  filter(grepl('P_jeeves', roi))

faces_cs <- jeeves_cs %>% 
  filter(grepl('face_patch', roi))

cbrs_cs <- jeeves_cs %>% 
  filter(grepl('CBR', roi))

rois = list(shapes, places, faces, cbrs)
rois_cs =list(shapes_cs, places_cs, faces_cs, cbrs_cs)
to_plot <- function(roi_set) {
  (roi_plot <- ggplot(roi_set, aes(x=roi, y=cross_mean)) +
    geom_col()+
    geom_errorbar(aes(x=roi, ymin=cross_mean-cross_var, ymax=cross_mean+cross_var)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
     #labs(title = (paste0("Shape to color decoding for ", roi_set$roi))) +
     theme_classic()
  )
}
lapply(rois, to_plot)
lapply(rois_cs, to_plot)
lapply(jeeves_cs_cbrs, to_plot)

(roi_plot <- ggplot(jeeves_cs_cbrs, aes(x=roi, y=cross_mean)) +
    geom_col()+
    geom_errorbar(aes(x=roi, ymin=cross_mean-cross_var, ymax=cross_mean+cross_var)) +
    geom_hline(yintercept=1/6, linetype = "dashed") +
    #labs(title = (paste0("Shape to color decoding for ", roi_set$roi))) +
    theme_classic()
)


# load data
testminipoa <- read.csv('/media/ssbeast/DATA/Users/Helen/20230904_2bfs_sc_poa.csv')

testminipoa <- testminipoa %>% 
  group_by(roi, frac) %>% 
  summarise(mean_decode = mean(cross_mean),
            mean_var = mean(cross_var))

roi_I_want <- testminipoa %>% 
  filter(roi == "rh_ant_CBR_jeeves")


poa_plot <- ggplot(roi_I_want, aes(x=frac, y=mean_decode)) +
  geom_point() +
  geom_linerange(aes(ymin=mean_decode-mean_var, ymax=mean_decode+mean_var)) +
  geom_hline(yintercept=1/6, linetype = "dashed") +
  #geom_smooth(method = "lm", formula = y ~ log(x))+
  scale_x_continuous(expand = c(0,0),
                     limits = c(0, 1.2),
                     breaks = seq(0, 1.2, 0.1)) +
  scale_y_continuous(expand = c(0,0),
                     limits = c(0, 0.3)) +
  labs(x = "percent of total data used to train and test",
       y = "decoding accuracy",
       title = "right hemi ant CBR") +
  theme_classic()
poa_plot







all_iters <- read.csv("/media/ssbeast/DATA/Users/KarthikKasi/finaldata.csv")

# original data has decoding accuracy for each fraction of the dataset for each iteration that that was run
power_analysis <- all_iters %>% 
  # for each roi, for each fraction of the dataset
  group_by(roi, frac) %>% 
  # get the mean id and cross decoding accuracies and errors
  summarise(mean_id = mean(roi_train),
            mean_cross = mean(roi_cross),
            std_id_decode = mean(train_var),
            std_cross_decode = mean(cross_var),
            ymax_id = mean_id+std_id_decode,
            ymin_id = mean_id-std_id_decode,
            ymax_cross = mean_cross+std_cross_decode,
            ymin_cross = mean_cross-std_cross_decode) %>% 
  # get rid of error columns, we have needed info in min max columns
  select(-c(std_id_decode, std_cross_decode)) %>% 
  # pivot everything long
  pivot_longer(cols = c(mean_id, mean_cross, ymax_id, ymin_id, ymax_cross, ymin_cross),
               names_to = c("meanorsd", "idorcross"),
               values_to = "value",
               names_sep = "_") %>%
  # pivot back
  pivot_wider(names_from = meanorsd, values_from = value)

# make dataframe for a single roi
RhAntCBR <- power_analysis %>% 
  filter(roi == "rh_ant_CBR_jeeves")

# make plot for a single roi
RhAntCBR_plot <- ggplot(RhAntCBR, aes(x=frac, y=mean, group=idorcross)) +
  geom_point(aes(color = idorcross)) +
  geom_linerange(aes(ymax=ymax, ymin=ymin, color = idorcross)) +
  geom_hline(yintercept=1/6, linetype = "dashed") +
  geom_smooth(aes(color = idorcross), method = "lm", formula = y ~ log(x))+
  scale_x_continuous(expand = c(0,0),
                     limits = c(0, 1.0),
                   breaks = seq(0, 1.0, 0.1)) +
  scale_y_continuous(expand = c(0,0),
                     limits = c(0, 0.5)) +
  labs(x = "percent of total data used to train and test",
       y = "decoding accuracy",
       title = "right hemi ant CBR") +
  theme_classic()
RhAntCBR_plot

# make dataframe with only cross-decoding
cross <- power_analysis %>% 
  filter(idorcross == "cross")

# quickly visualize the trajectories of all rois
quick_vis_all <- ggplot(cross, aes(x=frac, y=mean)) +
  geom_point(aes(color=roi)) +
  geom_smooth(aes(color=roi), se=FALSE) +
  geom_hline(yintercept=1/6, linetype = "dashed") 
quick_vis_all

# poa_data <- read.csv("/media/ssbeast/DATA/Users/KarthikKasi/finaldata.csv")
# poa <- poa_data %>% 
#   # for each roi, for each fraction of the dataset
#   group_by(roi, frac) %>% 
#   # get the mean id and cross decoding accuracies and errors
#   summarise(mean_id_decode = mean(roi_train),
#             mean_cross_decode = mean(roi_cross),
#             std_id_decode = mean(train_var),
#             std_cross_decode = mean(cross_var),
#             ymax_id = mean_id_decode+std_id_decode,
#             ymin_id = mean_id_decode-std_id_decode,
#             ymax_cross = mean_cross_decode+std_cross_decode,
#             ymin_cross = mean_cross_decode-std_cross_decode) 
# 
# poa_rh_ant_CBR_jeeves <- poa %>% 
#   filter(roi == "rh_ant_CBR_jeeves")
# 
# poa_rh_post_CBR2 <- poa_rh_post_CBR_jeeves %>% 
#   # rename columns to make pivot easier
#   mutate(mean_id = mean_id_decode,
#          mean_cross = mean_cross_decode) %>% 
#   select(-c(std_id_decode, std_cross_decode)) %>% 
#   # pivot everything long
#   pivot_longer(cols = c(mean_id, mean_cross, ymax_id, ymin_id, ymax_cross, ymin_cross),
#                names_to = c("meanorsd", "idorcross"),
#                values_to = "value",
#                names_sep = "_") %>% 
#   # pivot back
#   pivot_wider(names_from = meanorsd, values_from = value) %>% 
#   select(-c(mean_id_decode, mean_cross_decode))
# 
# better_plot <- ggplot(poa_rh_post_CBR2, aes(x=frac, y=mean, group=idorcross)) +
#   geom_point(aes(color = idorcross)) +
#   geom_linerange(aes(ymax=ymax, ymin=ymin, color = idorcross)) +
#   geom_hline(yintercept=0.166, linetype = "dashed") +
#   geom_smooth(aes(color = idorcross), method = "lm", formula = y ~ log(x))+
#   scale_x_continuous(expand = c(0,0),
#                      limits = c(0, 1.0),
#                    breaks = seq(0, 1.0, 0.1)) +
#   scale_y_continuous(expand = c(0,0),
#                      limits = c(0, 0.5)) +
#   labs(x = "percent of total data used to train and test",
#        y = "decoding accuracy") +
#   theme_classic()
# better_plot
# 
# post_CBR_plot <- ggplot(poa_rh_post_CBR_jeeves, aes(x=frac)) +
#   geom_point(aes(y = mean_id_decode), color="red") +
#   #geom_errorbar(aes(ymax=ymax_id, ymin=ymin_id), color="red") +
#   geom_point(aes(y = mean_cross_decode), color="blue") +
#   #geom_errorbar(aes(ymax=ymax_cross, ymin=ymin_cross), color="blue")+
#   geom_hline(yintercept=0.166, linetype = "dashed") +
#   ylim(0, .5) +
#   theme_classic()
# post_CBR_plot
