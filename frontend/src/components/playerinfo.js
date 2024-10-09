const PlayerInfo = ( {playerData} ) => {
  return (
    <div>
        { (!playerData || !playerData.stats) ? (
         (<></>)
        ) : (
        <div>
            <img className = 'image' src = {playerData.image_url} />
            <table className = 'table'>
                <thead>
                    <tr>
                        <th>Team</th>
                        <th>Season</th>
                        <th>Age</th>
                        <th>GP</th>
                        <th>MP</th>
                        <th>FG%</th>
                        <th>FT%</th>
                        <th>PTS</th>
                        <th>3PM</th>
                        <th>AST</th>
                        <th>REB</th>
                        <th>STL</th>
                        <th>BLK</th>
                        <th>TOV</th>
                    </tr>
                </thead>
                <tbody>
                    {playerData.stats.map((season) => {
                        return <tr>
                            <td>{season['team']}</td>
                            <td>{season['season']}</td>
                            <td>{season['age']}</td>
                            <td>{season['gp'].toFixed(0)}</td>
                            <td>{season['mp'].toFixed(1)}</td>
                            <td>{season['fg%'].toFixed(3).replace(/^0+/, '')}</td>
                            <td>{season['ft%'].toFixed(3).replace(/^0+/, '')}</td>
                            <td>{season['pts'].toFixed(1)}</td>
                            <td>{season['3p'].toFixed(1)}</td>
                            <td>{season['ast'].toFixed(1)}</td>
                            <td>{season['reb'].toFixed(1)}</td>
                            <td>{season['stl'].toFixed(1)}</td>
                            <td>{season['blk'].toFixed(1)}</td>
                            <td>{season['tov'].toFixed(1)}</td>
                        </tr>              
                        })}
                </tbody>
            </table>
        </div>
        ) 
        }
    </div>
  )

}

export default PlayerInfo