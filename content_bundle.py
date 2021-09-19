from seaborn import categorical
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.arrays.sparse import dtype
import numpy as np
import plotly.graph_objects as go
import plotly as py
import pyarrow as pa
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import datetime
import plotly.graph_objects as go
import chart_studio.plotly as py
import networkx.algorithms.bipartite as bipartite
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


# function to created table from file uploaded
def get_df(file):
  # get extension and read file
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(file)
  return df
def prepare_data(files,selectuser):

    gender = get_df(files[0])
    # gender = gender.drop(columns = ['VIP hay Free','Year of Birth','Age Range','Age','Region','Operation Systems','Total Listerning','Total Listening Time (Min)'])
    gender=gender.rename(columns={'UserID (PK)':'User_ID'})

    # lay data clustering
    cluster = get_df(files[1])
    cluster=cluster.rename(columns={'User_ID (FK)':'User_ID'})

    cluster = pd.merge(cluster,gender,on = "User_ID")
    # lay data tu listening
    listening = get_df(files[2])
    listening = listening.drop(columns=['Unnamed: 11'])
    listening.columns = ['PlaylistID (PK)', 'Playlist Name', 'Category', 'Sub Category',
        'Playlist Type', 'Playlist Duration (min)', 'Actual Duration (min)',
        'User_ID', 'Listening Time','Listening Date',  'Listening Datetime']
    paid_vip,free_vip,listening_high_paid=0,0,0
    if selectuser == 'Paid':
        paid_vip = cluster[cluster['MainCluster_Description'].isin(['New Paid Users','Potential Users','Loyal Users'])]
        paid_vip = paid_vip.drop(columns=['Type_user','MainCluster_ID'])
        paid_vip = pd.merge(paid_vip,listening,on='User_ID')
        paid_vip = paid_vip[paid_vip['Actual Duration (min)'] > 1]
        listening_high_paid=paid_vip[['User_ID','Playlist Name','PlaylistID (PK)','Actual Duration (min)','Listening Date','Category','Sub Category']].reset_index(drop=True)
    elif  selectuser == 'Free':
        free_vip = cluster[cluster['MainCluster_Description'].isin(['Loyal Users, Engaging Listeners','Loyal Users, Skimming Listeners','Potential, Engaging Listeners','Potential, Skimming Listeners'])]
        free_vip = free_vip.drop(columns=['Type_user','MainCluster_ID'])
        free_vip = pd.merge(free_vip,listening,on='User_ID')
        free_vip = free_vip[free_vip['Actual Duration (min)'] > 1]
        listening_high_paid=free_vip[['User_ID','Playlist Name','PlaylistID (PK)','Actual Duration (min)','Listening Date','Category','Sub Category']].reset_index(drop=True)
    return listening_high_paid,free_vip,paid_vip
def EDA(free_vip,paid_vip,selectuser):
    if selectuser == 'Free':
        user = free_vip
    elif selectuser == 'Paid':
        user = paid_vip

    selectcluster = st.selectbox("Chọn Cluster", user['MainCluster_Description'].unique())

    Cluster = user[user['MainCluster_Description'] == selectcluster]

    Cate = user[user['MainCluster_Description'] == selectcluster]
    # Cate = Cluster[Cluster['Category']==selectcluster]
    st.subheader('a. Top sách được nghe nhiều nhất của {}'.format(selectcluster))
    st.write(Cate.groupby('Playlist Name')['Actual Duration (min)'].sum().reset_index().sort_values(by='Actual Duration (min)').tail(10))

    st.subheader('b. So sánh theo info của user {}'.format(selectcluster))
    color_list = ['#FE5722','#FEC208','#3F51B5','#4BAF4F','#9C28B1','#029688','#FCED3A','#E91D64','#CDDC39','#FD9800','#2095F2']
    loyal = user[user['MainCluster_Description'] == selectcluster]
    gender = loyal.groupby(['Gender']).nunique()['User_ID']
    gender['Other'] = gender['no information']+gender['other']
    gender = gender.drop(index = ['no information','other'])
    gender.index = ['Female','Male','Other/NA']

    dura = loyal.groupby(['Category','Gender']).sum('Actual Duration (min)')['Actual Duration (min)'].unstack()
    dura['Other'] = dura['no information']+dura['other'].fillna(0)
    dura = dura.drop(columns = ['no information','other'])
    dura.columns = ['Female','Male','Other/NA']
    # f,m,o = (dura['Female'].sum(),dura['Male'].sum(),dura['Other/NA'].sum())
    cols = dura.columns
    chart = dura
    chart[cols] = dura[cols].div(dura[cols].sum(axis=0),axis = 1).multiply(100)
    chart = chart.transpose()
    col1, col2 = st.columns(2)

    with col1:
        ax = chart.plot.barh(stacked=True,color = color_list)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18),ncol=3, fancybox=False,prop={'size': 12.3})
        ax.set_xlabel('% Thời lượng nghe theo Category',fontsize = 16.4)
        # plt.title('% lượng nghe theo Category' , loc = 'bottom')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


    incate = loyal.groupby(['Sub Category','Gender']).sum('Actual Duration (min)')['Actual Duration (min)'].unstack()
    
    incate['Other'] = incate['no information']+incate['other'].fillna(0)
    incate = incate.drop(columns = ['no information','other'])
    incate.columns = ['Female','Male','Other/NA']
    cols = incate.columns
    chart = incate

    chart[cols] = incate[cols].div(incate[cols].sum(axis=0),axis = 1).multiply(100)
    chart = chart.transpose()

    cat80 = loyal.groupby(['Sub Category']).sum('Actual Duration (min)')['Actual Duration (min)']

    pv1= cat80.sort_values(ascending=False).reset_index()
    pv1=pv1.rename(columns = {'Actual Duration (min)':'Duration by segment'})
    #find number of top 80% subcategory
    T80=0
    for n in range(len(pv1)):
        if (1-T80/sum(loyal['Actual Duration (min)']))>=0.2:
            T80=T80+pv1['Duration by segment'].iloc[n]
        else:
            break
    dfc=pd.merge(loyal[['Sub Category','Gender','Actual Duration (min)']],pv1.head(n+1),on='Sub Category').sort_values('Duration by segment', ascending=False)
    dfc.drop(columns="Duration by segment")
    dfc = dfc.groupby(['Sub Category','Gender']).sum('Actual Duration (min)')['Actual Duration (min)'].unstack()
    dfc['Other'] = dfc['no information']+dfc['other'].fillna(0)
    dfc = dfc.drop(columns = ['no information','other'])
    dfc.columns = ['Female','Male','Other/NA']
    cols = dfc.columns
    chartdfc = dfc

    chartdfc[cols] = dfc[cols].div(dfc[cols].sum(axis=0),axis = 1).multiply(100)
    chartdfc = chartdfc.transpose()
    with col2:
        labels = 'Female', 'Male', 'Other/NA'
        sizes = [gender.loc['Female'], gender.loc['Male'], gender.loc['Other/NA']]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,textprops={'fontsize': 16})
        ax1.axis('equal') 
        ax1.set_xlabel('% User theo giới tính',fontsize = 15)
        ax1.xaxis.set_label_coords(0.5,-0.1)
        plt.show()
        st.pyplot()

    ax = chartdfc.plot.barh(stacked=True,color=color_list)
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),ncol=1, fancybox=False)
    ax.set_xlabel('% Thời lượng nghe của Top 80% Subcategory',fontsize = 12)
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()




def rank(x): return x['time_event'].rank(method='first').astype(int)
def get_next_event(x): return x['event_name'].shift(-1)
def get_time_diff(x): return x['time_event'].shift(-1) - x['time_event']
def update_source_target(user):
        try:
            source_index = output['nodes_dict'][user.name[1]]['sources_index'][output['nodes_dict']
                                                                            [user.name[1]]['sources'].index(user['event_name'].values[0])]

            target_index = output['nodes_dict'][user.name[1] + 1]['sources_index'][output['nodes_dict']
                                                                                [user.name[1] + 1]['sources'].index(user['next_event'].values[0])]

            if source_index in output['links_dict']:
                if target_index in output['links_dict'][source_index]:

                    output['links_dict'][source_index][target_index]['unique_users'] += 1
                    output['links_dict'][source_index][target_index]['avg_time_to_next'] += user['time_to_next'].values[0]
                else:

                    output['links_dict'][source_index].update({target_index:
                                                            dict(
                                                                {'unique_users': 1,
                                                                    'avg_time_to_next': user['time_to_next'].values[0]}
                                                            )
                                                            })
            else:

                output['links_dict'].update({source_index: dict({target_index: dict(
                    {'unique_users': 1, 'avg_time_to_next': user['time_to_next'].values[0]})})})
        except Exception as e:
            pass
def encode_data(datapoint):
    if datapoint <= 0:
        return 0
    if datapoint >= 1:
        return 1
def basket_content(high_paid_group,listening_high_paid,selectuser):
    high_paid_group=listening_high_paid.groupby(['Playlist Name']).agg({'Actual Duration (min)':'mean', 'User_ID': 'count'})
    high_paid_group=pd.DataFrame(high_paid_group).reset_index().sort_values('Actual Duration (min)')
    high_paid_group.sort_values('User_ID')
    if selectuser=='Paid':
        c=np.quantile(high_paid_group['User_ID'],0.99)
        high_paid_group_80=high_paid_group[(high_paid_group['User_ID']<=c)]

    # q=np.quantile(high_paid_group['Actual Duration (min)'],0.95)
    else:

        high_paid_group_80=high_paid_group.copy()
    high_paid_group_80.reset_index().sort_values('User_ID')
    list_80=high_paid_group_80['Playlist Name'].unique().tolist()
    # import pandas as pd

    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    listening_high_paid_=listening_high_paid[listening_high_paid['Playlist Name'].isin(list_80)]
    listening_high_paid_=listening_high_paid_[listening_high_paid_['Playlist Name'].str.contains('Muôn Kiếp')==False]
    listening_high_paid_group=listening_high_paid_.groupby(['User_ID','Playlist Name'])['PlaylistID (PK)']

    listening_high_paid_market = listening_high_paid_group.count().unstack().reset_index().fillna(0).set_index('User_ID')
    # listening_high_paid_market.columns
    listening_high_paid_market=listening_high_paid_market.applymap(encode_data)
    # listening_high_paid_market
    itemsets=apriori(listening_high_paid_market, min_support=0.01, use_colnames=True)
    # itemsets

    rules = association_rules(itemsets, metric="lift")
    rules=rules[rules['lift']>=1].sort_values('lift')
    # a['antecedents']=a['antecedents'].astype(str)
    rules["antecedents"].apply(lambda x: str(x))
    cols = ['antecedents','consequents']
    rules[cols] = rules[cols].applymap(lambda x: tuple(x))
    df_association_rules = (rules.explode('antecedents')
            .reset_index(drop=True)
            .explode('consequents')
            .reset_index(drop=True))
    return df_association_rules

files = st.file_uploader("Tải file user -cluster - listening ", type=['csv','xlsx','pickle'],accept_multiple_files=True)

if not files:
    st.warning('Upload file to continue')
else:
    # st.set_page_config(layout="wide")
    st.title('EXPLORE DATA')
    st.sidebar.markdown('## Thông tin')
    selectuser = st.sidebar.selectbox("Loại User", ['Free','Paid'])   

    list_data=prepare_data(files,selectuser)
    listening_high_paid=list_data[0]
    free_vip=list_data[1]
    paid_vip=list_data[2]

    st.subheader('1. Phân tích hành vi của user theo các nhóm cluster')
    EDA(free_vip,paid_vip,selectuser)

    st.subheader('2. Theo dõi hành trình của user')
    listen_hist=listening_high_paid.copy()
    listen_hist['event_name']=listen_hist['Category']
    listen_hist['event_type1']='listen'
    listen_hist['Listening Date']=pd.to_datetime(listen_hist['Listening Date'])
    listen_hist['time_event']=listen_hist['Listening Date']
    listen_hist=listen_hist.sort_values(by=['User_ID','time_event'],ascending=True)
    first_listen=listen_hist.groupby(['User_ID','event_name']).agg({'time_event': np.min}).reset_index().sort_values('User_ID')

    first_listen['event_type']='listen'
    list_user=first_listen['User_ID'].unique().tolist()

    data=first_listen.copy()
    data['time_event']=pd.to_datetime(data['time_event'])
    first=st.multiselect('Choosing Category',
                data['event_name'].unique().tolist())
    if not first:
        st.info('Chọn category')
    else:
        # df_types = pd.DataFrame(data.dtypes, columns=['Data Type'])
        # df_types
        grouped = data.groupby('User_ID')
        # st.write('print')
        # grouped
        data["rank_event"] = grouped.apply(rank).reset_index(0, drop=True)
        data["next_event"] = grouped.apply(
            lambda x: get_next_event(x)).reset_index(0, drop=True)

        data["time_to_next"] = 1

        data = data[data.rank_event < 9]


        data=data[data['event_name'].isin(first)]
        data=data[data['next_event'].isnull()==False].reset_index(drop=True)


        all_events = data.next_event.unique().tolist()
        # all_events
        # Create a set of colors that you'd like to use in your plot.
        palette = ['50BE97', 'E4655C', 'FCC865',
                'BFD6DE', '3E5066', '353A3E', 'E6E6E6']
        #  Here, I passed the colors as hex, but we need to pass it as RGB. This loop will do:
        for i, col in enumerate(palette):
            palette[i] = tuple(int(col[i:i+2], 16) for i in (0, 2, 4))

        # Append a Seaborn complementary palette to your palette in case you did not provide enough colors to style every event

        output = dict()
        output.update({'nodes_dict': dict()})

        i = 0
        for rank_event in data.rank_event.unique():  # For each rank of event...
            # Create a new key equal to the rank...
            output['nodes_dict'].update(
                {rank_event: dict()}
            )

            # Look at all the events that were done at this step of the funnel...
            all_events_at_this_rank = data[data.rank_event ==
                                        rank_event].next_event.unique().tolist()

            # Read the colors for these events and store them in a list...
            rank_palette = []
            for event in all_events_at_this_rank:
                rank_palette.append(tuple(palette[all_events.index(event)]))

            # Keep trace of the events' names, colors and indices.
            output['nodes_dict'][rank_event].update(
                {
                    'sources': all_events_at_this_rank,
                    'color': rank_palette,
                    'sources_index': range(i, i+len(all_events_at_this_rank))
                }
            )
            # Finally, increment by the length of this rank's available events to make sure next indices will not be chosen from existing ones
            i += len(output['nodes_dict'][rank_event]['sources_index'])
        # st.write('dfg')
        # output
        # Working on the links_dict

        output.update({'links_dict': dict()})

        # Group the DataFrame by User_ID and rank_event
        grouped = data.groupby(['User_ID', 'rank_event'])

        # Apply the function to your grouped Pandas object:
        grouped.apply(lambda user: update_source_target(user))
        targets = []
        sources = []
        values = []
        time_to_next = []

        for source_key, source_value in output['links_dict'].items():
            for target_key, target_value in output['links_dict'][source_key].items():
                sources.append(source_key)
                targets.append(target_key)
                values.append(target_value['unique_users'])
                time_to_next.append(str(pd.to_timedelta(
                    target_value['avg_time_to_next'] / target_value['unique_users'])).split('.')[0])  # Split to remove the milliseconds information

        labels = []
        colors = []
        for key, value in output['nodes_dict'].items():
            labels = labels + output['nodes_dict'][key]['sources']
            colors = colors + output['nodes_dict'][key]['color']

        for idx, color in enumerate(colors):
            colors[idx] = "rgb" + str(color) + ""

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                thickness=20,  # default is 20
                line=dict(color="black", width=1),
                label=labels,
                color=colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=time_to_next,
                hovertemplate='%{value} unique users went from %{source.label} to %{target.label}.<br />' +
                '<br />It took them %{label} in average.<extra></extra>',
            ))])

        fig.update_layout(autosize=True, title_text="Content_Journey",
                        font=dict(size=15), plot_bgcolor='white')
        st.plotly_chart(fig)

        st.subheader('B. Combo nội dung') 
        st.markdown('B1. Playlist nào đang "link" được nhiều nhất?')

        high_paid_group=listening_high_paid.groupby(['Playlist Name']).agg({'Actual Duration (min)':'mean', 'User_ID': 'count'})
        df_association_rules=basket_content(high_paid_group,listening_high_paid,selectuser)
        dff=df_association_rules[['antecedents','consequents']]
        dff.columns=['antecedents','Playlist Name']
        dff=dff.merge(listening_high_paid,how='left',on='Playlist Name')
        dff=dff[['antecedents','Playlist Name','Category','Sub Category']].drop_duplicates()
        dff_=dff.groupby(['antecedents','Category','Sub Category']).count().reset_index()
        # dff_[dff_['Playlist Name']>5]
        dff_pivot=dff_.pivot_table(index='antecedents',columns=['Category','Sub Category'],values='Playlist Name').reset_index()
        dff_pivot=dff_pivot.rename(columns={'antecedents':'Playlist Name'})
        dataa_df=dff_pivot.merge(listening_high_paid,how='left',on='Playlist Name')
        dataa_df=dataa_df.drop(columns=['Actual Duration (min)','Playlist Name','User_ID','PlaylistID (PK)','Listening Date']).drop_duplicates().reset_index(drop=True)
        dataa_df=dataa_df.replace(np.nan,0)
        dataa_df

        st.markdown('B2. Danh sách đề xuất theo playlist đã nghe')
        content=st.selectbox('Chọn playlist đã nghe',df_association_rules['antecedents'].unique())
        list_bundle= df_association_rules[df_association_rules['antecedents'].str.contains(content)]
        list_bundle=list_bundle[['antecedents','consequents','lift']].drop_duplicates(subset='consequents').sort_values('lift',ascending=False).reset_index(drop=True)
        # list_bundle
        G=nx.from_pandas_edgelist(list_bundle,source="antecedents",target='consequents')
        # st.write(G)
        r = list_bundle.copy()
        edges = []
        for idx, rr in r.iterrows():
                # rr['antecedents']
                edges.append((rr['antecedents'], rr['consequents']))   
        g = nx.DiGraph()
        g .add_edges_from(edges)
        plt.figure(figsize = (20, 10))
        nx.draw(g, with_labels = True,node_size = 5000, font_size = 20)
        plt.show()
        st.pyplot()




