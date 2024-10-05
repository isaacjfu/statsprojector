const PlayerInfo = ( {playerData} ) => {
  return (
    <div>
        <img src = {playerData.image_url} />
        { (!playerData || !playerData.stats) ? (
         (<></>)
        ) : (<table>
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
                        <td>{season['gp']}</td>
                        <td>{season['mp']}</td>
                        <td>{season['fg%']}</td>
                        <td>{season['ft%']}</td>
                        <td>{season['pts']}</td>
                        <td>{season['3p']}</td>
                        <td>{season['ast']}</td>
                        <td>{season['reb']}</td>
                        <td>{season['stl']}</td>
                        <td>{season['blk']}</td>
                        <td>{season['tov']}</td>
                    </tr>              
                    })}
            </tbody>
        </table>) 
        }
    </div>
  )

}

export default PlayerInfo