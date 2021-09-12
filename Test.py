# C:\Users\Nam Hung\OneDrive - DKSH\Documents\GitHub\Self-learn Project\Prime-data\Sample\Hackathon\Test.py
from seaborn import categorical
import streamlit as st
from streamlit_lottie import st_lottie
import pandas as pd
import json
import matplotlib.pyplot as plt
from pandas.core.arrays.sparse import dtype
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly as py
import plotly
import pyarrow as pa
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



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
def select(selectuser):
    if selectuser == 'Free':
        user = free_vip
    elif selectuser == 'Paid':
        user = paid_vip
    
    # Select Box
    selectcluster = st.sidebar.selectbox("Chọn Cluster", user['MainCluster_Description'].unique())
    # selectcate = st.selectbox("Category", user['Category'].unique())
    # selectsubcate = st.selectbox("Sub Category", user['Sub Category'].unique())

    Cluster = user[user['MainCluster_Description'] == selectcluster]

    col1, col2 = st.columns(2)

    Cate = user[user['MainCluster_Description'] == selectcluster]
    # Cate = Cluster[Cluster['Category']==selectcluster]
    st.subheader('a. Top sách được nghe nhiều nhất của {}'.format(selectcluster))
    st.write(Cate.groupby('Playlist Name')['Actual Duration (min)'].sum().sort_values(by='Actual Duration (min)').tail(10).reset_index())

    st.subheader('b. So sánh theo info của user {}'.format(selectcluster))
    col1, col2 = st.columns(2)
    loyal = user[user['MainCluster_Description'] == selectcluster]
    dura = loyal.groupby(['Category','Gender']).sum('Actual Duration (min)')['Actual Duration (min)'].unstack()
    dura['Other'] = dura['no information']+dura['other'].fillna(0)
    dura = dura.drop(columns = ['no information','other'])
    dura.columns = ['Female','Male','Other']
    cols = dura.columns
    chart = dura
    chart[cols] = dura[cols].div(dura[cols].sum(axis=0),axis = 1).multiply(100)
    chart = chart.transpose()
    with col1:
        ax = chart.plot.barh(stacked=True)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    incate = loyal.groupby(['Sub Category','Gender']).sum('Actual Duration (min)')['Actual Duration (min)'].unstack()
    
    incate['Other'] = incate['no information']+incate['other'].fillna(0)
    incate = incate.drop(columns = ['no information','other'])
    incate.columns = ['Female','Male','Other']
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
    dfc.columns = ['Female','Male','Other']
    cols = dfc.columns
    chartdfc = dfc

    chartdfc[cols] = dfc[cols].div(dfc[cols].sum(axis=0),axis = 1).multiply(100)
    chartdfc = chartdfc.transpose()
    with col2:
        ax = chartdfc.plot.barh(stacked=True)
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


st.write("Hãy tải lên 1 file định dạng .csv or .xlsx để bắt đầu xem báo cáo")
st.write('Thứ tự lần lượt là: user_processing - cluster - listening')
files = st.file_uploader("Tải file", type=['csv','xlsx','pickle'],accept_multiple_files=True)


if not files:
    st.warning('Upload file to continue')
else:
# st.set_page_config(layout="wide")

    st.sidebar.markdown('## Thông tin')

    selectuser = st.sidebar.selectbox("Loại User", ['Free','Paid'])

    gender = get_df(files[0])
    gender = gender.drop(columns = ['VIP hay Free','Year of Birth','Age Range','Age','Region','Operation Systems','Total Listerning','Total Listening Time (Min)'])
    gender.columns = ['User_ID','Gender','Regristration Date']
    # gender
    # lay data clustering
    cluster = get_df(files[1])
    cluster.columns = ['User_ID', 'Type_user', 'MainCluster_ID','MainCluster_Description']
    cluster = pd.merge(cluster,gender,on = "User_ID")

    # lay data tu listening
    listening = get_df(files[2])
    listening = listening.drop(columns=['Unnamed: 11'])
    listening.columns = ['PlaylistID (PK)', 'Playlist Name', 'Category', 'Sub Category',
        'Playlist Type', 'Playlist Duration (min)', 'Actual Duration (min)',
        'User_ID', 'Listening Time','Listening Date',  'Listening Datetime']
    # lay data order
    # order = get_df(files[3])
    # order = order.drop(columns=['Unnamed: 11'])
    # order
    # Free
    free = cluster[cluster['Type_user'] == 'free']
    free_vip = free[free['MainCluster_Description'].isin(['Loyal Users, Engaging Listeners','Loyal Users, Skimming Listeners','Potential, Engaging Listeners','Potential, Skimming Listeners'])]
    free_vip = free_vip.drop(columns=['Type_user','MainCluster_ID'])
    free_vip = pd.merge(free_vip,listening,on='User_ID')
    free_vip = free_vip[free_vip['Actual Duration (min)'] > 1]

    # Paid
    paid = cluster[cluster['Type_user']=='paid']
    paid_vip = cluster[cluster['MainCluster_Description'].isin(['New Paid Users','Potential Users','Loyal Users'])]
    paid_vip = paid_vip.drop(columns=['Type_user','MainCluster_ID'])
    paid_vip = pd.merge(paid_vip,listening,on='User_ID')
    paid_vip = paid_vip[paid_vip['Actual Duration (min)'] > 1]

        
    st.subheader('1. Phân tích hành vi của user theo các nhóm cluster')
    
    select(selectuser)

    #B. JOURNEY CUSTOMER ANALYSIS
    st.subheader('2. Theo dõi hành trình của user')
    # order_vip=pd.merge(paid_vip,order,on='User_ID')
    


    # if st.button('finding journey of customer'):

    if selectuser == 'Paid':
        high_paid_df=paid_vip[['User_ID','Regristration Date']].reset_index(drop=True)
        listening_high_paid=paid_vip[['User_ID','Playlist Name','PlaylistID (PK)','Actual Duration (min)','Listening Date','Category']].reset_index(drop=True)
        # order_high_paid=order     
        # high_paid_df
        # listening_high_paid
    elif  selectuser == 'Free':
        high_paid_df=free_vip[['User_ID','Regristration Date']].reset_index(drop=True)
        listening_high_paid=free_vip[['User_ID','Playlist Name','PlaylistID (PK)','Actual Duration (min)','Listening Date','Category']].reset_index(drop=True)
        # order_high_paid=order     
        # high_paid_df
        # listening_high_paid

    import numpy as np
    listen_hist=listening_high_paid.copy()
    listen_hist['event_name']=listen_hist['Category']
    listen_hist['event_type1']='listen'
    listen_hist['Listening Date']=pd.to_datetime(listen_hist['Listening Date'])
    listen_hist['time_event']=listen_hist['Listening Date']
    listen_hist=listen_hist.sort_values(by=['User_ID','time_event'],ascending=True)
    first_listen=listen_hist.groupby(['User_ID','event_name']).agg({'time_event': np.min}).reset_index().sort_values('User_ID')

    first_listen['event_type']='listen'
    list_user=first_listen['User_ID'].unique().tolist()
    # first_listen


    # first_install=high_paid_df.copy()
    # first_install['event_name']='Regristration Date'
    # first_install['event_type']='Regristration Date'
    # first_install['Regristration Date']=pd.to_datetime(first_install['Regristration Date'])
    # first_install['time_event']=first_install['Regristration Date']
    # first_install=first_install.drop(columns='Regristration Date',axis=0)
    # first_install=first_install.drop_duplicates(subset='User_ID')
    # first_install=first_install[first_install['User_ID'].isin(list_user)]
    # first_install=first_install.drop_duplicates()
        # first_install
    # data=first_install.append(first_listen).sort_values(by=['User_ID','event_type'	,'time_event'],ascending=True)
    # data=data.drop_duplicates()
    # data=data[data['User_ID']==40843]
    data=first_listen.copy()
    data['time_event']=pd.to_datetime(data['time_event'])
    first=st.multiselect('Choosing Category',
                data['event_name'].unique().tolist())
    if first:
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
        # data[data['rank_event'] == 1].event_name.unique()
        # list_error=data[(data['rank_event']!=1) &(data['event_type']=='install app')]
        # len(list_error['User_ID'].unique().tolist())
        # # data['time_to_next']=data['time_to_next'].astype(str)
        # data=data[data['User_ID'].isin(list_error)==False]



        import datetime
        data=data[data['event_name'].isin(first)]
        data=data[data['next_event'].isnull()==False].reset_index(drop=True)
        # data=data[data['time_event'].month==second]
        # data


        # import seaborn as sns
        import pandas as pd
        import plotly.graph_objects as go
        import chart_studio.plotly as py
        import plotly
        # data=data[:20]

        # Working on the nodes_dict

        all_events = data.next_event.unique().tolist()
        # all_events
        # Create a set of colors that you'd like to use in your plot.
        palette = ['50BE97', 'E4655C', 'FCC865',
                'BFD6DE', '3E5066', '353A3E', 'E6E6E6']
        #  Here, I passed the colors as hex, but we need to pass it as RGB. This loop will do:
        for i, col in enumerate(palette):
            palette[i] = tuple(int(col[i:i+2], 16) for i in (0, 2, 4))

        # Append a Seaborn complementary palette to your palette in case you did not provide enough colors to style every event
        # complementary_palette = sns.color_palette(
        #     "deep", len(all_events) - len(palette))
        # if len(complementary_palette) > 0:
        #     palette.extend(complementary_palette)

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

        # Define a function to read the souces, targets, values and time from event to next_event:


        
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

    import numpy as np
    # listening_high_paid
    high_paid_group=listening_high_paid.groupby(['Playlist Name']).agg({'Actual Duration (min)':'mean', 'User_ID': 'count'})
    high_paid_group=pd.DataFrame(high_paid_group).reset_index().sort_values('Actual Duration (min)')
    high_paid_group.sort_values('User_ID')
    # c=np.quantile(high_paid_group['User_ID'],0.99)
    q=np.quantile(high_paid_group['Actual Duration (min)'],0.95)
    high_paid_group_80=high_paid_group[(high_paid_group['Actual Duration (min)']>=q)]
    high_paid_group_80.reset_index().sort_values('User_ID')
    list_80=high_paid_group_80['Playlist Name'].unique().tolist()
    # import pandas as pd

    from mlxtend.frequent_patterns import apriori
    from mlxtend.frequent_patterns import association_rules
    listening_high_paid_=listening_high_paid[listening_high_paid['Playlist Name'].isin(list_80)]
    # listening_high_paid_[listening_high_paid_['Playlist Name'].str.contains('Muôn kiếp')]
    listening_high_paid_group=listening_high_paid_.groupby(['User_ID','Playlist Name'])['PlaylistID (PK)']

    listening_high_paid_market = listening_high_paid_group.count().unstack().reset_index().fillna(0).set_index('User_ID')
    # listening_high_paid_market.columns
    listening_high_paid_market=listening_high_paid_market.applymap(encode_data)
    # listening_high_paid_market
    itemsets=apriori(listening_high_paid_market, min_support=0.01, use_colnames=True)
    # itemsets

    rules = association_rules(itemsets, metric="lift",min_threshold=.5)
    rules=rules[rules['lift']>1].sort_values('lift')
    # a['antecedents']=a['antecedents'].astype(str)
    rules["antecedents"].apply(lambda x: str(x))
    cols = ['antecedents','consequents']
    rules[cols] = rules[cols].applymap(lambda x: tuple(x))
    df_association_rules = (rules.explode('antecedents')
            .reset_index(drop=True)
            .explode('consequents')
            .reset_index(drop=True))
    content=st.selectbox('Chọn cuốn sách đã đọc',df_association_rules['antecedents'].unique())

    list_bundle= df_association_rules[df_association_rules['antecedents'].str.contains(content)]
    list_bundle=list_bundle[['consequents','lift']].drop_duplicates(subset='consequents').sort_values('lift',ascending=False).reset_index(drop=True)
    list_bundle
